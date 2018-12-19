import socket
import threading
import matplotlib.pyplot as plt
import numpy as np
import traning
import time


n = 0
x = 0
old = 0


def on_new_connection(client, result):
    global x, n, old
    print(time.time() - old)
    all_msg = ""

    if result >= 0:
        client.send(str(int(result)).encode())

    while True:
        try:
            msg = client.recv(1024 * 200).decode("utf-8")
        except:
            client.close()
            print("断开连接")
            return 1

        if "!" in msg:
            msg = msg.replace("!", "")
            all_msg += msg
        else:
            all_msg += msg
            print(all_msg)

        if all_msg != "" and all_msg[0] == "]":
            all_msg = ""

        if "clear" in all_msg:
            plt.clf()
            all_msg = ""

        if "|" in all_msg:
            line = np.transpose(eval(all_msg.split("|")[0]))
            plt.plot(line[0], line[1])
            all_msg = all_msg[all_msg.index("|") + 1:]

        if "end" in all_msg:
            old = time.time()
            plt.axis("off")
            plt.savefig("save.png", dpi=20)
            plt.clf()
            return 0


listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.bind(('0.0.0.0', 10086))
listener.listen(5)
print('Waiting for connect...')


while True:
    client_executor, addr = listener.accept()
    print('Accept new connection')
    t = threading.Thread(target=traning.evaluate_one_image, args=(on_new_connection, client_executor))
    t.start()