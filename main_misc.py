import os
import torchvision as tv

def merge_image():
    dir_arr = [
        'ckpt/2023-05-10_imagenet256_stsp4/generated_s05_Cls10-20_bl',   # baseline, 5 steps
        'ckpt/2023-05-10_imagenet256_stsp4/generated_s10_Cls10-20_bl',
        'ckpt/2023-05-10_imagenet256_stsp4/generated_s15_Cls10-20_bl',
        'ckpt/2023-05-10_imagenet256_stsp4/generated_s20_Cls10-20_bl',
        'ckpt/2023-05-10_imagenet256_stsp4/generated_s05_Cls10-20_vubo', # vubo, 5 steps
        'ckpt/2023-05-10_imagenet256_stsp4/generated_s10_Cls10-20_vubo',
        'ckpt/2023-05-10_imagenet256_stsp4/generated_s15_Cls10-20_vubo',
        'ckpt/2023-05-10_imagenet256_stsp4/generated_s20_Cls10-20_vubo',
    ]
    dir_out = 'ckpt/2023-05-10_imagenet256_stsp4/generated_Cls10-20_all'
    if not os.path.exists(dir_out):
        print(f"os.makedirs({dir_out})")
        os.makedirs(dir_out)
    f_list = os.listdir(dir_arr[0])
    f_list.sort()  # file name list
    for fn in f_list:
        file_arr = [os.path.join(d, fn) for d in dir_arr]
        img_arr = [tv.io.read_image(f) for f in file_arr]
        img_arr = [i.float() for i in img_arr]
        img_arr = [i / 255.0 for i in img_arr]
        file_out = os.path.join(dir_out, fn)
        print(f"saving image: {file_out}")
        tv.utils.save_image(img_arr, file_out, nrow=4)


if __name__ == "__main__":
    merge_image()
