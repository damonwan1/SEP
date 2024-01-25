if [ ! -d "./result" ]; then
    mkdir ./result
fi

data_folder="../../../data"
output_folder="./result"
datasets=$(ls $data_folder)

for dataset in $datasets
do
    full_path="$data_folder/$dataset"
    out_path="_$dataset"
    
    echo "Running model for dataset: $dataset"
    
    python -u scompressorTwo.py \
    --input_dir $full_path \
    --prefix $out_path > "result/$dataset.log"

    echo "Finished model for dataset: $dataset"
done

echo "All datasets have been processed successfully."
