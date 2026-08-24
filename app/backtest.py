def walk_forward_splits(series, initial_train, horizon, step, mode="expanding"):  # mode = expanding or sliding 
    n = len(series)
    train_end = initial_train
    train_start = 0
    

    while train_end + horizon <= n:
        if mode == "sliding":
                train_start = train_end - initial_train
                
        train_idx = series.index[train_start : train_end]
        test_idx = series.index[train_end : train_end+horizon]

        yield train_idx, test_idx

        train_end += step
