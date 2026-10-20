idf_component_register(SRCS "gesture_recognition.cpp" "gesture_model.cc"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES tflm mpu6050 driver esp_wifi esp_http_server nvs_flash)

                   

                    
