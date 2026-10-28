/*
    MJPEG writer with a small HTTP router on top.
    Based on https://github.com/JPery/MJPEGWriter/
*/

#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <signal.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

#define PORT        unsigned short
#define SOCKET    int
#define HOSTENT  struct hostent
#define SOCKADDR    struct sockaddr
#define SOCKADDR_IN  struct sockaddr_in
#define ADDRPOINTER  unsigned int*
#define INVALID_SOCKET -1
#define SOCKET_ERROR   -1
#define TIMEOUT_M       200000
#define NUM_CONNECTIONS 10

using namespace cv;
using namespace std;

struct clientFrame {
    uchar* outbuf;
    int outlen;
    int client;
};

struct clientPayload {
    void* context;
    clientFrame cf;
};

class MJPEGWriter {
public:
    /* Full HTTP response (headers + body) for a non-stream request.
       Return an empty string to fall back to the built-in 404. */
    typedef std::string (*HttpHandler)(const std::string &method, const std::string &path,
                                       const std::string &body);

private:
    SOCKET sock;
    fd_set master;
    int timeout;
    int quality;
    std::vector<int> clients;
    pthread_t thread_listen, thread_write;
    pthread_mutex_t mutex_client = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mutex_cout = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mutex_writer = PTHREAD_MUTEX_INITIALIZER;
    std::vector<uchar> lastJpeg;
    uint64_t frameSeq = 0;
    int port;
    HttpHandler httpHandler = NULL;

    int _write(int sock, char *s, int len) {
        if (len < 1) { len = strlen(s); }
        int retval = ::send(sock, s, len, MSG_NOSIGNAL);
        return retval;
    }

    static void* listen_Helper(void* context) {
        ((MJPEGWriter *)context)->Listener();
        return NULL;
    }

    static void* writer_Helper(void* context) {
        ((MJPEGWriter *)context)->Writer();
        return NULL;
    }

    static void* clientWrite_Helper(void* payload) {
        void* ctx = ((clientPayload *)payload)->context;
        struct clientFrame cf = ((clientPayload *)payload)->cf;
        ((MJPEGWriter *)ctx)->ClientWrite(cf);
        return NULL;
    }

public:
    MJPEGWriter(int port = 0)
        : sock(INVALID_SOCKET), timeout(TIMEOUT_M), quality(90), port(port) {
        signal(SIGPIPE, SIG_IGN);
        FD_ZERO(&master);
    }

    ~MJPEGWriter() { release(); }

    bool release() {
        if (sock != INVALID_SOCKET) shutdown(sock, 2);
        sock = (INVALID_SOCKET);
        return false;
    }

    bool open() {
        sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        int one = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

        SOCKADDR_IN address;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_family = AF_INET;
        address.sin_port = htons(port);
        if (::bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR) {
            cerr << "error : couldn't bind sock " << sock << " to port " << port << "!" << endl;
            return release();
        }
        if (listen(sock, NUM_CONNECTIONS) == SOCKET_ERROR) {
            cerr << "error : couldn't listen on sock " << sock << " on port " << port << " !" << endl;
            return release();
        }
        FD_SET(sock, &master);
        return true;
    }

    bool isOpened() { return sock != INVALID_SOCKET; }

    void setHttpHandler(HttpHandler h) { httpHandler = h; }

    void start() {
        pthread_mutex_lock(&mutex_writer);
        pthread_create(&thread_listen, NULL, this->listen_Helper, this);
        pthread_create(&thread_write, NULL, this->writer_Helper, this);
    }

    void stop() {
        this->release();
        /* Рвём клиентские соединения: зависшая отправка кадра иначе не даст
           завершиться потоку writer. */
        pthread_mutex_lock(&mutex_client);
        for (size_t i = 0; i < clients.size(); i++) {
            ::shutdown(clients[i], 2);
            ::close(clients[i]);
        }
        clients.clear();
        pthread_mutex_unlock(&mutex_client);
        /* Потоки не дожидаемся: клиент, который перестал читать, может
           держать send() до таймаута, а приложению нужно освободить VB-пулы
           и завершиться. Потоки умрут вместе с процессом. */
        pthread_detach(thread_listen);
        pthread_detach(thread_write);
    }

    void writeJPEG(const void *data, size_t len) {
        if (data == NULL || len == 0) return;
        pthread_mutex_lock(&mutex_writer);
        lastJpeg.assign((const uchar*)data, (const uchar*)data + len);
        frameSeq++;
        pthread_mutex_unlock(&mutex_writer);
    }

    void write(Mat frame) {
        if (frame.empty()) return;
        std::vector<uchar> outbuf;
        std::vector<int> params;
        params.push_back(CV_IMWRITE_JPEG_QUALITY);
        params.push_back(quality);
        imencode(".jpg", frame, outbuf, params);
        if (!outbuf.empty()) writeJPEG(outbuf.data(), outbuf.size());
    }

private:
    void Listener();
    void Writer();
    void ClientWrite(clientFrame &cf);
};
