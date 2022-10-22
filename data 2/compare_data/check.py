import os

REAL_PATH = os.path.join('real_data')
FAKE_PATH = os.path.join('fake_data')

if __name__ == '__main__':
    real_names = os.listdir(REAL_PATH)
    fake_names = os.listdir(FAKE_PATH)

    print(len(real_names))
    print(len(fake_names))

    for i in range(len(real_names)):
        real_names[i] = real_names[i].replace('MS_', '')
        real_names[i] = real_names[i].replace('MT_', '')
        real_names[i] = real_names[i].replace('-', '')
        real_names[i] = real_names[i].replace('_', '')
    
    for i in range(len(fake_names)):
        fake_names[i] = fake_names[i].replace('_back', '')
        fake_names[i] = fake_names[i].replace('_front', '')
        fake_names[i] = fake_names[i].replace('-', '')
        fake_names[i] = fake_names[i].replace('_', '')

    print(real_names[10])
    print(fake_names[10])