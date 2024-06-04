# for all files in configs/game_configs, run the game with the config file
for user_file in configs/user_configs/*; do
    for assistant_file in configs/assistant_configs/*; do
        echo "Running game with user config: $user_file and assistant config: $assistant_file"
        ~/anaconda3/envs/pragma/bin/python hai_1stage_game.py --user_config $user_file --assistant_config $assistant_file
    echo "\n----------------------------------------------------\n"
    done
done