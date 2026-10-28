/*
    MJPEG writer with a tiny HTTP router.
    Based on https://github.com/JPery/MJPEGWriter/
*/

#include "MJPEGWriter.h"
#include <fstream>
#include <algorithm>
#include <cctype>

namespace {

/* Case-insensitive search for an HTTP header value. */
std::string header_value(const std::string &req, const char *name) {
    std::string low;
    low.reserve(req.size());
    for (size_t i = 0; i < req.size(); i++) low += (char)tolower((unsigned char)req[i]);
    std::string key = name;
    for (size_t i = 0; i < key.size(); i++) key[i] = (char)tolower((unsigned char)key[i]);
    size_t p = low.find("\r\n" + key + ":");
    if (p == std::string::npos) return "";
    p = req.find(':', p + 2);
    if (p == std::string::npos) return "";
    size_t e = req.find("\r\n", p);
    std::string v = req.substr(p + 1, e == std::string::npos ? std::string::npos : e - p - 1);
    while (!v.empty() && (v[0] == ' ' || v[0] == '\t')) v.erase(v.begin());
    while (!v.empty() && (v[v.size() - 1] == ' ' || v[v.size() - 1] == '\t')) v.erase(v.size() - 1);
    return v;
}

} // namespace

void MJPEGWriter::Listener() {
    const std::string mjpeg_header =
        "HTTP/1.0 200 OK\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n\r\n";

    fd_set rread;
    SOCKET maxfd;
    this->open();
    pthread_mutex_unlock(&mutex_writer);
    while (true) {
        rread = master;
        struct timeval to = {0, timeout};
        maxfd = sock + 1;
        if (sock == INVALID_SOCKET) return;
        int sel = select(maxfd, &rread, NULL, NULL, &to);
        if (sel <= 0) continue;
        if (!FD_ISSET(sock, &rread)) continue;

        int addrlen = sizeof(SOCKADDR);
        SOCKADDR_IN address = {0};
        SOCKET client = accept(sock, (SOCKADDR *)&address, (socklen_t *)&addrlen);
        if (client == SOCKET_ERROR) {
            cerr << "error : couldn't accept connection on sock " << sock << " !" << endl;
            return;
        }

        /* Read the request head; browsers send it in one go. A short timeout
           keeps a half-open or preconnected socket from blocking the accept
           loop. */
        struct timeval rtv = {0, 300000};
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));
        /* Отправка кадра тоже ограничена по времени: клиент, который перестал
           читать (например, закрыл вкладку), иначе вешает поток writer, а
           через него — остановку приложения. */
        struct timeval stv = {0, 3000000};
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof(stv));
        std::string req;
        char buf[2048];
        size_t hdr_end = std::string::npos;
        while (req.size() < 16384) {
            int n = recv(client, buf, sizeof(buf), 0);
            if (n <= 0) break;
            req.append(buf, (size_t)n);
            hdr_end = req.find("\r\n\r\n");
            if (hdr_end != std::string::npos) break;
        }
        if (hdr_end == std::string::npos) {
            ::shutdown(client, 2);
            ::close(client);
            continue;
        }

        std::string method, path;
        {
            size_t sp1 = req.find(' ');
            size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : req.find(' ', sp1 + 1);
            if (sp1 != std::string::npos && sp2 != std::string::npos) {
                method = req.substr(0, sp1);
                path = req.substr(sp1 + 1, sp2 - sp1 - 1);
            }
        }
        size_t q = path.find('?');
        if (q != std::string::npos) path = path.substr(0, q);

        size_t content_len = 0;
        {
            std::string cl = header_value(req.substr(0, hdr_end), "content-length");
            if (!cl.empty()) content_len = (size_t)strtoul(cl.c_str(), NULL, 10);
            if (content_len > 8192) content_len = 8192;
        }
        size_t body_start = hdr_end + 4;
        while (req.size() - body_start < content_len) {
            int n = recv(client, buf, sizeof(buf), 0);
            if (n <= 0) break;
            req.append(buf, (size_t)n);
        }
        std::string body = req.substr(body_start, content_len);

        if (path == "/stream" || path == "/mjpg" || path == "/video") {
            pthread_mutex_lock(&mutex_client);
            _write(client, (char *)mjpeg_header.data(), (int)mjpeg_header.size());
            clients.push_back(client);
            pthread_mutex_unlock(&mutex_client);
            continue;
        }

        std::string resp;
        if (httpHandler != NULL) resp = httpHandler(method, path, body);
        if (resp.empty())
            resp = "HTTP/1.0 404 Not Found\r\nContent-Type: text/plain; charset=utf-8\r\n"
                   "Connection: close\r\nContent-Length: 10\r\n\r\nnot found\n";
        size_t off = 0;
        while (off < resp.size()) {
            int n = _write(client, (char *)resp.data() + off, (int)(resp.size() - off));
            if (n <= 0) break;
            off += (size_t)n;
        }
        ::shutdown(client, 2);
        ::close(client);
    }
}

void MJPEGWriter::Writer() {
    pthread_mutex_lock(&mutex_writer);
    pthread_mutex_unlock(&mutex_writer);
    const int milis2wait = 16666;
    uint64_t sent = 0;
    while (this->isOpened()) {
        std::vector<uchar> outbuf;
        uint64_t seq;
        pthread_mutex_lock(&mutex_writer);
        seq = frameSeq;
        if (seq != sent) outbuf = lastJpeg;
        pthread_mutex_unlock(&mutex_writer);
        if (seq == sent) {
            usleep(2000);
            continue;
        }
        sent = seq;
        if (outbuf.empty()) continue;

        pthread_mutex_lock(&mutex_client);
        int num_connected_clients = clients.size();
        pthread_mutex_unlock(&mutex_client);
        if (!num_connected_clients) {
            usleep(milis2wait);
            continue;
        }
        pthread_t threads[NUM_CONNECTIONS];
        int count = 0;

        int outlen = outbuf.size();

        pthread_mutex_lock(&mutex_client);
        std::vector<int>::iterator begin = clients.begin();
        std::vector<int>::iterator end = clients.end();
        pthread_mutex_unlock(&mutex_client);
        std::vector<clientPayload *> payloads;
        for (std::vector<int>::iterator it = begin; it != end; ++it, ++count) {
            if (count >= NUM_CONNECTIONS) break;
            struct clientPayload *cp =
                new clientPayload({(MJPEGWriter *)this, {outbuf.data(), outlen, *it}});
            payloads.push_back(cp);
            pthread_create(&threads[count], NULL, &MJPEGWriter::clientWrite_Helper, cp);
        }
        for (; count > 0; count--) {
            pthread_join(threads[count - 1], NULL);
            delete payloads.at(count - 1);
        }
        usleep(1000);
    }
}

void MJPEGWriter::ClientWrite(clientFrame &cf) {
    stringstream head;
    head << "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: " << cf.outlen
         << "\r\n\r\n";
    string string_head = head.str();
    pthread_mutex_lock(&mutex_client);
    _write(cf.client, (char *)string_head.c_str(), (int)string_head.size());
    int n = _write(cf.client, (char *)(cf.outbuf), cf.outlen);
    if (n < cf.outlen) {
        /* клиент не читает — отключаем его, иначе он держит writer */
        std::vector<int>::iterator it = find(clients.begin(), clients.end(), cf.client);
        if (it != clients.end()) {
            cerr << "kill client " << cf.client << endl;
            clients.erase(std::remove(clients.begin(), clients.end(), cf.client));
            ::shutdown(cf.client, 2);
            ::close(cf.client);
        }
    }
    pthread_mutex_unlock(&mutex_client);
    pthread_exit(NULL);
}
