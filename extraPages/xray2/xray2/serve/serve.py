from http.server import *
import json, numpy as np, cv2, os


# class Handler(BaseHTTPRequestHandler):

class BaseHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kw):

        self.images = np.zeros((8,512,512,3),dtype=np.uint8)
        self.encImages = [cv2.imencode('.jpg', img)[1].tobytes() for img in self.images]
        super().__init__(*args, **kw)

    def get_regular_file(self, path, prefix='./client/public'):
        if path[0] == '/': path = path[1:]
        self.path = os.path.join(prefix, path)
        print('modify to', self.path)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        print('post', self.path)

        N = 1

        length = int(self.headers.get('Content-Length') or 0)
        body = {}
        if length > 0:
            body = self.rfile.read(length)
            body = json.loads(body)
            N = int(body.get('N', 1))
            print('have body', body)


        try:
            if self.path.startswith('/skeleton'):
                self.send_response(200)
                self.send_header('Content-Encoding', 'application/json')
                self.end_headers()

                d = self.get_skeletons(body);
                d = json.dumps(d).encode('ascii')
                print(f' - Returning {len(d)/1024/1024:3.1f}Mb response json.')

                self.wfile.write(d)

        except KeyboardInterrupt:
        # except:
            self.send_error(500)

    def do_GET(self):
        print('get', self.path)

        try:
            '''
            if self.path.startswith('/skeleton'):
                self.send_response(200)
                self.send_header('Content-Encoding', 'application/json')
                self.end_headers()

                d = { 'skeletons': {
                        'first': {
                            'indices': [0,1, 1,2],
                            'jointNames': ['foot', 'hip', 'chest'],
                            'T': 2,
                            'N': 2,
                            'positions': [
                                -.1,0,0, -.1,1,0, .5,2,0,
                                .1,0,0, .1,1,0, .5,2,0,

                                -.1,0,.1, -.1,1,.1, .5,2,.1,
                                .1,0,.1, .1,1,.1, .5,2,.1,
                            ],
                        }
                        }}
                self.wfile.write(json.dumps(d).encode('ascii'))
            '''

            if self.path.startswith('/image'):
                _,_,i = self.path.split('/')
                self.send_response(200)
                self.send_header('Content-Encoding', 'image/jpeg')
                self.end_headers()
                self.wfile.write(self.encImages[int(i)])

            elif self.path == '' or self.path == '/':
                return self.get_regular_file('index.html')
            else:
                return self.get_regular_file(self.path)

        except KeyboardInterrupt:
        # except:
            self.send_error(500)


# def run(server_class=ThreadingHTTPServer, handler_class=Handler):
def run(handler_class=BaseHandler, server_class=HTTPServer):
    server_address = ('', 3001)
    print(' - Starting server with addr spec', server_address)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

if __name__ == '__main__':
    run()
