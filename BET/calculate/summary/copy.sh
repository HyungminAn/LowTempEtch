for i in $(ls -d ../run/*/); do
    mol=$(basename ${i})
    cp ${i}/output_250.png ./output_${mol}.png
    echo ${mol} Done
done

tar -zcvf result.tar.gz *.png
