# for all files in configs/game_configs, run the game with the config file
for file in configs/game_configs/mmlu_clinical/*; do
    echo "Running game with config file: $file"
    ~/anaconda3/envs/pragma/bin/python game.py --config_file $file
    echo "\n----------------------------------------------------\n"
done
```