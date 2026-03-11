import math
import time
import os
import sys

def clear():
    os.system("cls" if os.name == "nt" else "clear")

width = 80
height = 40

t = 0

try:
    for _ in range(2):
        clear()
        for y in range(height):
            for x in range(width):
                cx = x - width / 2
                cy = y - height / 2

                distance = math.sqrt(cx**2 + cy**2)
                angle = math.atan2(cy, cx)

                value = math.sin(distance * 0.3 - t + angle)

                if value > 0.5:
                    char = "*"
                elif value > 0:
                    char = "+"
                elif value > -0.5:
                    char = "."
                else:
                    char = " "

                sys.stdout.write(char)
            sys.stdout.write("\n")

        sys.stdout.flush()
        t += 0.15
        time.sleep(0.05)

except KeyboardInterrupt:
    clear()
    print("Animation stopped.")
