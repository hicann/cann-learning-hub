#ifndef CROBOT_CONTROLLER_H
#define CROBOT_CONTROLLER_H

#include "crobot/controller_callbacks.h"
#include "crobot/listener.h"
#include "crobot/swsr_queue.h"
#include "crobot/message/request.h"
#include "CSerialPort/SerialPort.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>

namespace crobot {

class Controller {
private:
    itas109::CSerialPort sp_;
    crobot::Listener listener_;
    Controller_Callbacks& callbacks_;
    SWSR_Queue<uint8_t> data_queue_;

    bool thread_end_ = false;
    std::thread base_com_thread_;

    // 写串口移到独立线程，避免定时器/cmd_vel 回调在 writeData() 上阻塞 executor
    std::queue<std::vector<uint8_t>> write_queue_;
    std::mutex write_mutex_;
    std::condition_variable write_cv_;
    bool write_thread_end_ = false;
    std::thread write_thread_;
    void write_thread_func();

public:
    Controller(Controller_Callbacks& cbs);
    ~Controller();

    void init(const char* port_name,
              itas109::BaudRate baudrate,
              itas109::Parity parity,
              itas109::DataBits databits,
              itas109::StopBits stopbits,
              itas109::FlowControl flow_control);
    bool open();

    // receive
    void receive_data(uint8_t* data, uint32_t len);
    void process_response(const Response& resp);
    void base_com_func();

    // send（入队后立即返回；实际写由 write_thread_ 执行）
    void send_request(const Request& req);

private:
    void write(const Request& req);
};

} // namespace crobot

#endif // CROBOT_CONTROLLER_H
