#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "mpu6050.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "lwip/inet.h"  


#define DATA_COLLECTION_MODE 0

#define WIFI_SSID      "4G-D0BD"
#define WIFI_PASSWORD  "12345678"

#define I2C_MASTER_SCL_IO          9
#define I2C_MASTER_SDA_IO          8
#define I2C_MASTER_NUM             I2C_NUM_0
#define I2C_MASTER_FREQ_HZ         400000
#define MPU6050_ADDR               0x68

#define MODEL_INPUT_SIZE           100
#define DATA_DIM                   6
#define BUFFER_SIZE                MODEL_INPUT_SIZE * DATA_DIM

static uint8_t tensor_arena[64 * 1024] __attribute__((aligned(16)));
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

static mpu6050_handle_t mpu = NULL;
static float data_buffer[BUFFER_SIZE];

#define CLASS_CIRCLE   0
#define CLASS_X        1
#define CLASS_NONE     2
#define CLASS_INVALID  3
#define NUM_CLASSES    4

#define COOLDOWN_MS          1500

static int64_t last_output_time = 0;

static const float ALPHA = 0.98f;
static const float DT = 0.02f;
static const float RAD2DEG = 57.2958f;

static httpd_handle_t g_server = NULL;
static const char *TAG = "WEB";

// ================= WebSocket 客户端 fd 列表 =================
#define MAX_WS_CLIENTS 8
static int  g_ws_fds[MAX_WS_CLIENTS];
static bool g_ws_fd_used[MAX_WS_CLIENTS];

static void add_ws_client(int fd) {
    for (int i = 0; i < MAX_WS_CLIENTS; i++) {
        if (!g_ws_fd_used[i]) {
            g_ws_fd_used[i] = true;
            g_ws_fds[i] = fd;
            ESP_LOGI(TAG, "注册 WebSocket 客户端: fd=%d, slot=%d", fd, i);
            return;
        }
    }
    ESP_LOGW(TAG, "WebSocket 客户端槽位已满");
}

static void remove_ws_client(int fd) {
    for (int i = 0; i < MAX_WS_CLIENTS; i++) {
        if (g_ws_fd_used[i] && g_ws_fds[i] == fd) {
            g_ws_fd_used[i] = false;
            g_ws_fds[i] = -1;
            ESP_LOGI(TAG, "移除 WebSocket 客户端: fd=%d, slot=%d", fd, i);
            return;
        }
    }
}

// ================= 核心：向所有 WebSocket 客户端发文本 =================
static void send_ws_text(const char* msg) {
    if (g_server == NULL) return;

    httpd_ws_frame_t ws_pkt;
    memset(&ws_pkt, 0, sizeof(ws_pkt));
    ws_pkt.payload = (uint8_t*)msg;
    ws_pkt.len = strlen(msg);
    ws_pkt.type = HTTPD_WS_TYPE_TEXT;

    for (int i = 0; i < MAX_WS_CLIENTS; i++) {
        if (!g_ws_fd_used[i]) continue;
        esp_err_t ret = httpd_ws_send_frame_async(g_server, g_ws_fds[i], &ws_pkt);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "发送失败 fd=%d ret=%d，移除该客户端", g_ws_fds[i], (int)ret);
            remove_ws_client(g_ws_fds[i]);
        }
    }
}

// ================= 同时输出到串口和 WebSocket =================
static void log_both(const char* fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    printf("%s", buf);
    send_ws_text(buf);
}
// ============================================================


#if !DATA_COLLECTION_MODE

void setup_tflm() {
    printf("Step 1: Loading model (FP32, 4 classes)...\n");
    const tflite::Model* model = tflite::GetModel(gesture_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch\n");
        return;
    }
    printf("Model loaded OK\n");

    printf("Step 2: Creating resolver...\n");
    static tflite::MicroMutableOpResolver<22> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddExpandDims();
    resolver.AddMean();
    resolver.AddRelu();
    resolver.AddAveragePool2D();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    printf("Resolver created OK\n");

    printf("Step 3: Creating interpreter...\n");
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));
    interpreter = &static_interpreter;

    printf("Step 4: Allocating tensors...\n");
    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        printf("AllocateTensors failed: %d\n", status);
        return;
    }
    printf("arena used: %d bytes\n", interpreter->arena_used_bytes());

    input = interpreter->input(0);
    output = interpreter->output(0);
    if (input == nullptr || output == nullptr) {
        printf("Input/output null!\n");
        return;
    }

    printf("TFLM initialized OK\n");
}

void recognize_gesture() {
    if (input == nullptr || output == nullptr) return;

    static int log_idx = 0;

    static float world_buffer[BUFFER_SIZE];
    float pitch = 0.0f, roll = 0.0f;
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        float ax = data_buffer[i * DATA_DIM + 0];
        float ay = data_buffer[i * DATA_DIM + 1];
        float az = data_buffer[i * DATA_DIM + 2];
        float gx = data_buffer[i * DATA_DIM + 3];
        float gy = data_buffer[i * DATA_DIM + 4];
        float gz = data_buffer[i * DATA_DIM + 5];

        float acc_pitch = atan2f(-ax, sqrtf(ay*ay + az*az)) * RAD2DEG;
        float acc_roll  = atan2f(ay, az) * RAD2DEG;
        pitch = ALPHA * (pitch + gy * DT) + (1 - ALPHA) * acc_pitch;
        roll  = ALPHA * (roll  + gx * DT) + (1 - ALPHA) * acc_roll;

        float cp = cosf(pitch/RAD2DEG), sp = sinf(pitch/RAD2DEG);
        float cr = cosf(roll/RAD2DEG),  sr = sinf(roll/RAD2DEG);

        world_buffer[i * DATA_DIM + 0] = cr*ax - sr*az;
        world_buffer[i * DATA_DIM + 1] = sr*sp*ax + cp*ay + cr*sp*az;
        world_buffer[i * DATA_DIM + 2] = -sr*cp*ax + sp*ay + cr*cp*az;
        world_buffer[i * DATA_DIM + 3] = cr*gx - sr*gz;
        world_buffer[i * DATA_DIM + 4] = sr*sp*gx + cp*gy + cr*sp*gz;
        world_buffer[i * DATA_DIM + 5] = -sr*cp*gx + sp*gy + cr*cp*gz;
    }

    float max_energy = 0;
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        float gx = world_buffer[i * DATA_DIM + 3];
        float gy = world_buffer[i * DATA_DIM + 4];
        float gz = world_buffer[i * DATA_DIM + 5];
        float energy = gx*gx + gy*gy + gz*gz;
        if (energy > max_energy) max_energy = energy;
    }

    if (max_energy < 15000.0f) {
        return;
    }

    float* input_data = input->data.f;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        input_data[i] = world_buffer[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) return;

    float* output_data = output->data.f;
    float probs[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = output_data[i];
    }

    int best_id = 0;
    float best_prob = probs[0];
    float second_prob = 0.0f;

    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > best_prob) {
            second_prob = best_prob;
            best_prob = probs[i];
            best_id = i;
        } else if (probs[i] > second_prob) {
            second_prob = probs[i];
        }
    }

    float gyro_mag[MODEL_INPUT_SIZE];
    float mag_mean = 0;
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        float gx = world_buffer[i * DATA_DIM + 3];
        float gy = world_buffer[i * DATA_DIM + 4];
        float gz = world_buffer[i * DATA_DIM + 5];
        gyro_mag[i] = sqrtf(gx*gx + gy*gy + gz*gz);
        mag_mean += gyro_mag[i];
    }
    mag_mean /= MODEL_INPUT_SIZE;

    int zcr = 0;
    for (int i = 1; i < MODEL_INPUT_SIZE; i++) {
        if ((gyro_mag[i-1] - mag_mean) * (gyro_mag[i] - mag_mean) < 0) {
            zcr++;
        }
    }

    log_idx++;
    const char* gesture_name;
    if (best_id == CLASS_CIRCLE) gesture_name = "圆";
    else if (best_id == CLASS_X) gesture_name = "叉";
    else if (best_id == CLASS_NONE) gesture_name = "静止";
    else gesture_name = "无效";

    // ★ 修改点：识别结果行改用 log_both
    log_both("#%d 能量:%.0f | 圆:%.2f 叉:%.2f 静:%.2f 无效:%.2f | 判定:%s (优势:%.2f) ZCR:%d\n",
             log_idx, max_energy,
             probs[0], probs[1], probs[2], probs[3],
             gesture_name,
             best_prob - second_prob,
             zcr);

    int64_t now = (int64_t)xTaskGetTickCount() * portTICK_PERIOD_MS;
    if (now - last_output_time < COOLDOWN_MS) return;

    if (best_prob > 0.5f && (best_prob - second_prob) > 0.25f) {
        if (best_id == CLASS_CIRCLE) {
            log_both(">>> 圆 <<<\n");     // ★ 修改点
            last_output_time = now;
        } else if (best_id == CLASS_X) {
            log_both(">>> 叉 <<<\n");     // ★ 修改点
            last_output_time = now;
        }
    }
}

#endif


// ================= WebSocket handler =================

static esp_err_t ws_handler(httpd_req_t *req) {
    if (req->method == HTTP_GET) {
        int fd = httpd_req_to_sockfd(req);
        add_ws_client(fd);
        return ESP_OK;
    }
    return ESP_OK;
}

static httpd_uri_t ws_uri = {
    .uri        = "/ws",
    .method     = HTTP_GET,
    .handler    = ws_handler,
    .user_ctx   = NULL,
    .is_websocket = true
};

// ================= 根页面（终端风格日志） =================
static esp_err_t root_get_handler(httpd_req_t *req) {
    const char *html =
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>ESP32-S3 手势识别</title>"
        "<style>"
        "html,body{margin:0;padding:0;background:#000;color:#c9d1d9;"
        "  font-family:Consolas,monospace;height:100%;overflow:hidden;}"
        "#big{height:60vh;display:flex;align-items:center;justify-content:center;"
        "  font-size:28vh;font-weight:900;color:#30363d;background:#000;"
        "  transition:all 0.2s;letter-spacing:0.1em;}"
        "#big.circle{color:#3fb950;text-shadow:0 0 80px #3fb950,0 0 30px #3fb950;}"
        "#big.cross{color:#f85149;text-shadow:0 0 80px #f85149,0 0 30px #f85149;}"
        "#log{height:38vh;overflow-y:scroll;padding:10px 20px;font-size:16px;"
        "  background:#000;border-top:2px solid #30363d;}"
        ".line{padding:2px 0;white-space:pre-wrap;word-break:break-all;}"
        ".trigger{color:#f0c674;}"
        ".action{color:#56d4dd;}"
        ".result{color:#c9d1d9;}"
        ".skip{color:#6e7681;}"
        ".connected{color:#58a6ff;font-style:italic;}"
        "</style></head><body>"
        "<div id='big'>等待手势...</div>"
        "<div id='log'></div>"
        "<script>\n"
        "const ws=new WebSocket('ws://'+location.host+'/ws');\n"
        "const log=document.getElementById('log');\n"
        "const big=document.getElementById('big');\n"
        "let resetTimer=null;\n"
        "function addLine(text,cls){\n"
        "  const d=document.createElement('div');\n"
        "  d.className='line '+(cls||'result');\n"
        "  d.textContent=text;\n"
        "  log.appendChild(d);\n"
        "  if(log.children.length>300)log.removeChild(log.firstChild);\n"
        "  log.scrollTop=log.scrollHeight;\n"
        "}\n"
        "function showBig(symbol,text,cls){\n"
        "  big.textContent=symbol+' '+text;\n"
        "  big.className=cls;\n"
        "  if(resetTimer)clearTimeout(resetTimer);\n"
        "  resetTimer=setTimeout(()=>{big.textContent='等待手势...';big.className='';},3000);\n"
        "}\n"
        "ws.onopen=()=>addLine('[连接已建立，等待手势...]','connected');\n"
        "ws.onmessage=(e)=>{\n"
        "  const text=e.data;\n"
        "  let cls='result';\n"
        "  if(text.startsWith('[Trigger]'))cls='trigger';\n"
        "  else if(text.startsWith('[Action]'))cls='action';\n"
        "  else if(text.startsWith('[Skip]'))cls='skip';\n"
        "  else if(text.indexOf('>>> 圆')>=0){cls='circle';showBig('○','圆','circle');}\n"
        "  else if(text.indexOf('>>> 叉')>=0){cls='cross';showBig('✕','叉','cross');}\n"
        "  addLine(text.trimEnd(),cls);\n"
        "};\n"
        "ws.onclose=()=>addLine('[连接断开，请刷新页面重连]','skip');\n"
        "</script></body></html>";

    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}


static httpd_uri_t root_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = root_get_handler,
    .user_ctx  = NULL
};

// ================= 启动服务器 =================

void start_webserver(void) {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_open_sockets = 4;
    config.lru_purge_enable = true;
    config.stack_size = 8192;

    if (httpd_start(&g_server, &config) == ESP_OK) {
        httpd_register_uri_handler(g_server, &root_uri);
        httpd_register_uri_handler(g_server, &ws_uri);
        ESP_LOGI(TAG, "HTTP + WebSocket 服务器已启动，端口: %d", config.server_port);
    } else {
        ESP_LOGE(TAG, "HTTP 服务器启动失败");
    }
}

// ================= WiFi =================

void wifi_event_handler(void* arg, esp_event_base_t event_base,
                        int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi 断开，正在重连...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "获取到 IP 地址: " IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "请在浏览器中访问: http://" IPSTR, IP2STR(&event->ip_info.ip));
    }
}
void wifi_init_sta(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // ===== 静态 IP 配置 =====
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();

    esp_netif_dhcpc_stop(sta_netif);   // 关掉 DHCP 客户端

    esp_netif_ip_info_t ip_info;
    IP4_ADDR(&ip_info.ip,      192, 168, 43, 217);   // ESP32 的固定 IP
    IP4_ADDR(&ip_info.gw,      192, 168, 43, 1);     // 手机热点网关
    IP4_ADDR(&ip_info.netmask, 255, 255, 255, 0);
    esp_netif_set_ip_info(sta_netif, &ip_info);
    // ========================

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL, NULL));

    wifi_config_t wifi_config = {};
    strncpy((char*)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char*)wifi_config.sta.password, WIFI_PASSWORD, sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi 初始化完成，正在连接 %s ...", WIFI_SSID);
    ESP_LOGI(TAG, "固定 IP: 192.168.43.217");
}


// ================= 主任务 =================

void mpu6050_task(void *pvParameters) {
    mpu = mpu6050_create(I2C_MASTER_NUM, MPU6050_ADDR);
    if (mpu == NULL) { printf("MPU6050 fail\n"); vTaskDelete(NULL); }
    mpu6050_config(mpu, ACCE_FS_2G, GYRO_FS_250DPS);
    mpu6050_wake_up(mpu);
    vTaskDelay(pdMS_TO_TICKS(100));

#if DATA_COLLECTION_MODE
    printf("Data collection mode: ON\n");
    while (1) {
        mpu6050_acce_value_t acce;
        mpu6050_gyro_value_t gyro;
        if (mpu6050_get_acce(mpu, &acce) == ESP_OK && 
            mpu6050_get_gyro(mpu, &gyro) == ESP_OK) {
            printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                   -acce.acce_x, acce.acce_y, acce.acce_z,
                   gyro.gyro_x, gyro.gyro_y, gyro.gyro_z);
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
#else
    setup_tflm();
    printf("Gesture recognition started (dynamic window)\n");

    static float ring_buffer[200 * DATA_DIM];
    static int ring_pos = 0;
    static int total_frames = 0;
    static int action_started = 0;
    static int frames_since_trigger = 0;
    static int action_end_frames = 0;
    static float action_max_energy = 0;

    while (1) {
        mpu6050_acce_value_t acce;
        mpu6050_gyro_value_t gyro;

        if (mpu6050_get_acce(mpu, &acce) == ESP_OK && 
            mpu6050_get_gyro(mpu, &gyro) == ESP_OK) {
            ring_buffer[ring_pos * DATA_DIM + 0] = -acce.acce_x;
            ring_buffer[ring_pos * DATA_DIM + 1] = acce.acce_y;
            ring_buffer[ring_pos * DATA_DIM + 2] = acce.acce_z;
            ring_buffer[ring_pos * DATA_DIM + 3] = gyro.gyro_x;
            ring_buffer[ring_pos * DATA_DIM + 4] = gyro.gyro_y;
            ring_buffer[ring_pos * DATA_DIM + 5] = gyro.gyro_z;

            ring_pos = (ring_pos + 1) % 200;
            total_frames++;

            float gx = gyro.gyro_x;
            float gy = gyro.gyro_y;
            float gz = gyro.gyro_z;
            float energy = gx*gx + gy*gy + gz*gz;

            if (!action_started && energy > 20000.0f && total_frames > 120) {
                action_started = 1;
                frames_since_trigger = 0;
                action_end_frames = 0;
                action_max_energy = 0;
                log_both("[Trigger] 动作开始, energy=%.0f\n", energy);  // ★ 修改
            }

            if (action_started) {
                frames_since_trigger++;
                
                if (energy > action_max_energy) {
                    action_max_energy = energy;
                }
                
                if (energy > 10000.0f) {
                    action_end_frames = 0;
                } else {
                    action_end_frames++;
                }

                int action_done = 0;
                if (action_end_frames > 30 && frames_since_trigger > 50) {
                    action_done = 1;
                }
                if (frames_since_trigger >= 180) {
                    action_done = 1;
                }

                if (action_done) {
                    int action_length = frames_since_trigger - action_end_frames;

                    int active_frames = 0;
                    int sample_frames = 0;
                    int start_check = ring_pos - frames_since_trigger;
                    if (start_check < 0) start_check += 200;
                    
                    for (int k = 0; k < action_length; k++) {
                        int idx = (start_check + k) % 200;
                        float egx = ring_buffer[idx * DATA_DIM + 3];
                        float egy = ring_buffer[idx * DATA_DIM + 4];
                        float egz = ring_buffer[idx * DATA_DIM + 5];
                        if (egx*egx + egy*egy + egz*egz > 5000.0f) {
                            active_frames++;
                        }
                        sample_frames++;
                    }
                    float active_ratio = (sample_frames > 0) ? 
                                         ((float)active_frames / sample_frames) : 0;

                    int should_skip = 0;
                    const char* skip_reason = "";

                    if (action_length < 20) {
                        should_skip = 1;
                        skip_reason = "段太短";
                    } else if (action_length < 35 && action_max_energy < 100000 && active_ratio < 0.5f) {
                        should_skip = 1;
                        skip_reason = "疑似拿起";
                    }

                    if (should_skip) {
                        log_both("[Skip] %s, 段长=%d, 最大能量=%.0f, 活跃率=%.2f\n",
                                 skip_reason, action_length, action_max_energy, active_ratio);  // ★ 修改
                        action_started = 0;
                        vTaskDelay(pdMS_TO_TICKS(1500));
                        continue;
                    }

                    log_both("[Action] 段长=%d帧, 最大能量=%.0f, 活跃率=%.2f\n",
                             action_length, action_max_energy, active_ratio);  // ★ 修改

                    if (action_length < 20) action_length = 20;

                    int start_offset = ring_pos - frames_since_trigger - 10;
                    if (start_offset < 0) start_offset += 200;

                    int total_extract = action_length + 20;
                    if (total_extract > 200) total_extract = 200;
                    if (total_extract < 20) total_extract = 20;

                    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
                        float pos = (float)i * (total_extract - 1) / (MODEL_INPUT_SIZE - 1);
                        int idx = (int)pos;
                        float frac = pos - idx;
                        for (int a = 0; a < DATA_DIM; a++) {
                            int s1 = (start_offset + idx) % 200;
                            int s2 = (start_offset + idx + 1) % 200;
                            float v1 = ring_buffer[s1 * DATA_DIM + a];
                            float v2 = ring_buffer[s2 * DATA_DIM + a];
                            data_buffer[i * DATA_DIM + a] = v1 + frac * (v2 - v1);
                        }
                    }

                    recognize_gesture();

                    action_started = 0;
                    vTaskDelay(pdMS_TO_TICKS(1500));
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
#endif
}


extern "C" void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init_sta();
    start_webserver();

    i2c_config_t conf;
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = I2C_MASTER_SDA_IO;
    conf.scl_io_num = I2C_MASTER_SCL_IO;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = I2C_MASTER_FREQ_HZ;
    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
    xTaskCreate(mpu6050_task, "mpu6050_task", 32768, NULL, 5, NULL);
}

