# Introduction
This is our project for Reinforcement Learning with PyBoy with different levels of abstraction, where we trained agents to play Kirby, with different levels of understanding of the game: In-Memory data, Tiles and Pixel data. 

# Requirements
- Python3.7 on Windows
- Python3+ on Linux
- PyBoy (https://github.com/Baekalfen/PyBoy)
- SDL2
    - Ubuntu: __`sudo apt install libsdl2-dev`__
    - Fedora: __`sudo dnf install SDL2-devel`__
    - macOS: __`brew install sdl2`__
    - Windows: PyBoy guide https://github.com/Baekalfen/PyBoy/wiki/Installation#windows-10-64-bit Download link https://www.libsdl.org/download-2.0.php
- For package requirements see requirements.txt
- GameBoy ROM files for Super Mario Land or Kirby's Dream Land (place these in /games) 
  - Filename must be "Kirby_Dream_Land" and "Super_Mario_Land" respectively

# Run with Python
To run from source, first install dependencies:
- __`pip3 install -r requirements.txt`__

Then, run:
- __`python3 main.py`__


# Based on
PyBoy-RL - https://github.com/lixado/PyBoy-RL/raw/main/README/report.pdf
