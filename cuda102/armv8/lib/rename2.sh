VERSHORT=10
VER=${VERSHORT}.2.1.89


for filename in *.so; do
    if [ -f ${filename}.${VER} ]; then
    	rm $filename
    	ln -s ${filename}.${VER} ${filename}
    fi
    if [ -f ${filename}.${VERSHORT} ]; then
    	rm ${filename}.${VERSHORT}
    	ln -s ${filename}.${VER} ${filename}.${VERSHORT}
    fi
done

