#!/bin/bash

root_dir=/media/khuongthinh/Me-Family-Work/caffe-data/trainingData
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset=test

  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in te-hm-m-01
  do
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/PNGImages\//g" $img_file
    sed -i "s/$/.png/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi
