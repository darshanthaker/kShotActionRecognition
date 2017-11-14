#!/bin/bash

mkdir -p data
if [ ! -e data/kinetics_readme.txt ]; then
    wget https://deepmind.com/documents/67/kinetics_readme.txt -P data/
fi
if [ ! -d data/kinetics_train ]; then
    wget https://deepmind.com/documents/66/kinetics_train.zip -P data/
    unzip data/kinetics_train.zip -d data/
fi
if [ ! -d data/kinetics_val ]; then
    wget https://deepmind.com/documents/65/kinetics_val.zip -P data/
    unzip data/kinetics_val.zip -d data/
fi
if [ ! -d data/kinetics_test ]; then
    wget https://deepmind.com/documents/81/kinetics_test.zip -P data/
    unzip data/kinetics_test.zip -d data/
fi

mkdir -p packages
if [ ! -e packages/youtube-dl ]; then
    curl -L https://yt-dl.org/downloads/latest/youtube-dl -o packages/youtube-dl
    chmod +x packages/youtube-dl
fi

export DIR=$(pwd)
mkdir -p $DIR/ffmpeg_sources
cd $DIR/ffmpeg_sources
curl -O -L http://www.nasm.us/pub/nasm/releasebuilds/2.13.01/nasm-2.13.01.tar.bz2
curl -O -L http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
curl -O -L https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
if [ ! -d nasm-2.13.01 ]; then
    echo "Installing NASM..."
    tar xjf nasm-2.13.01.tar.bz2
    cd nasm-2.13.01
    ./autogen.sh
    ./configure --prefix="$DIR/ffmpeg_build" --bindir="$DIR/bin"
    make
    make install
    cd ../
    echo
fi

if [ ! -d yasm-1.3.0 ]; then
    echo "Installing YASM..."
    tar xzf yasm-1.3.0.tar.gz
    cd yasm-1.3.0
    ./configure --prefix="$DIR/ffmpeg_build" --bindir="$DIR/bin" 
    make
    make install
    cd ../
    echo
fi

if [ ! -d x264 ]; then
    echo "Installing x264 encoder..."
    git clone --depth 1 http://git.videolan.org/git/x264
    cd x264
    PATH="$DIR/bin:$PATH" PKG_CONFIG_PATH="$DIR/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$DIR/ffmpeg_build" --bindir="$DIR/bin" --enable-static
    PATH="$DIR/bin:$PATH" make
    PATH="$DIR/bin:$PATH" make install
    cd ../
fi

if [ ! -d ffmpeg ]; then
    echo "Installing FFMPEG..."
    tar xjf ffmpeg-snapshot.tar.bz2
    cd ffmpeg
    PATH="$DIR/bin:$PATH" PKG_CONFIG_PATH="$DIR/ffmpeg_build/lib/pkgconfig" ./configure \
          --prefix="$DIR/ffmpeg_build" \
          --pkg-config-flags="--static" \
          --extra-cflags="-I$DIR/ffmpeg_build/include" \
          --extra-ldflags="-L$DIR/ffmpeg_build/lib" \
          --extra-libs=-lpthread \
          --bindir="$DIR/bin" \
          --enable-gpl \
          --enable-libx264
    PATH="$DIR/bin:$PATH" make
    PATH="$DIR/bin:$PATH" make install
    cd ../
    echo
fi
cd ../
