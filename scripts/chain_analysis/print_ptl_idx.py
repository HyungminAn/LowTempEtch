import sys


def main():
    numbers = sys.argv[1:]
    numbers = " || ".join([f"ParticleIndex == {num}" for num in numbers])
    print(numbers)


if __name__ == "__main__":
    main()
