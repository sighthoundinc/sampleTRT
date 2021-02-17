for filename in *.so; do
    if [ -f ${filename}.10.0.326 ]; then
    	rm $filename
    	ln -s ${filename}.10.0.326 ${filename}
    fi
    if [ -f ${filename}.10.0 ]; then
    	rm ${filename}.10.0
    	ln -s ${filename}.10.0.326 ${filename}.10.0
    fi
done

