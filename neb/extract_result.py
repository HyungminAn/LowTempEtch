import os
import sys
import yaml
from ase.io import read, write

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv} <path_yaml> <neb_traj>")
        sys.exit()

    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)

    path_traj = sys.argv[2]
    n_images = inputs['neb']['n_images'] + 2
    path_output = inputs['neb']['path_result']
    images = read(path_traj, index=':')

    dst = "result"
    os.makedirs(dst, exist_ok=True)
    images = images[-n_images:]
    write(f"{dst}/{path_output}", images, format='extxyz')
    for idx, image in enumerate(images):
        write(f"{dst}/POSCAR_{idx}", image, format='vasp')


if __name__ == "__main__":
    main()
