This project uses the [stb library](https://github.com/nothings/stb) and is therefore licensed accordingly.

```
make
./pathtracer
```

mp4 generated using

```
cd cat_gif
ffmpeg -framerate 24 -i frame_%03d.png -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" cat.mp4
```
