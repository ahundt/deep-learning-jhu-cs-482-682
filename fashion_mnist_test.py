

    if args.test == 'models':
        # TODO create all models and at least call their init and forward functions
        model = Net()
        model = Q13UltimateNet()
        batch_size = 1000
        test_batch_size = 1000
        epochs_to_run = 1
    else: