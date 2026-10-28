#ifndef FACE_WEBUI_H
#define FACE_WEBUI_H

#include <string>

/* Full HTTP response for a non-stream request; empty -> 404. */
std::string web_handle(const std::string &method, const std::string &path,
                       const std::string &body);

#endif
