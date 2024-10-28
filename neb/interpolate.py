import sys

from ase.io import read, write
from ase.mep import NEB


def main():
    if len(sys.argv) != 2:
        print("Usage: python interpolate.py <path_images>")
        sys.exit(1)

    path_images = sys.argv[1]
    multiple_factor = 3
    images = read(path_images, index=':')

    image_tot = []
    image_tot += [images[0]]
    for idx, (image_i, image_j) in enumerate(zip(images[:-1], images[1:])):
        interpolate = [image_i]
        interpolate += [image_i.copy() for i in range(multiple_factor-1)]
        interpolate.append(image_j)
        neb = NEB(interpolate)
        neb.interpolate()
        image_tot += neb.images[1:]
        print(f"Interpolating between {idx} and {idx+1}")

    write("result.extxyz", image_tot, format="extxyz")


if __name__ == "__main__":
    main()
