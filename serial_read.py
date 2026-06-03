import serial

ser = serial.Serial(port='COM15',
                    baudrate=115200,
                    timeout=8
)

f = open('data.txt', 'w')

while True:
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('cp949').rstrip()
            print(line)
            f.write(line + '\n')
    except KeyboardInterrupt:
        break

f.close()