"""All simulations run for this project are defined here"""

if __name__ == "__main__":
    from fl_simulator import Server, Simulation, FedAvg, GIANT
    from constants import CONFIG

    # This is For FedAVG.
    # For GIANT we must divide this by two
    NUM_ROUNDS = 50
    # N_BYZ = 5

    # Global parameters, constant through different simulations
    CONFIG["N_CLIENTS"] = 20
    CONFIG["MAX_CLIENT_ITERS"] = None

    # ###### 1 - Run the simulation with the GIANT strategy, NO ATTACKERS ######
    # CONFIG["NUM_ROUNDS"] = NUM_ROUNDS // 2
    # CONFIG["N_BYZANTINE_CLIENTS"] = 0
    # CONFIG["MODEL_NAME"] = f"GIANT_{CONFIG['N_CLIENTS']}C"

    # strategy = GIANT(CONFIG["MAX_CLIENT_ITERS"])
    # server = Server(CONFIG, reduce_op="mean")

    # sim = Simulation(
    #     config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    # )
    # sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    ###### 2 - Run the same simulation, but with the FedAvg strategy ######
    # CONFIG["NUM_ROUNDS"] = NUM_ROUNDS
    # CONFIG["N_BYZANTINE_CLIENTS"] = 0
    # CONFIG["MODEL_NAME"] = f"FedAvg_{CONFIG['N_CLIENTS']}C"

    # strategy = FedAvg(CONFIG["MAX_CLIENT_ITERS"])
    # server = Server(CONFIG, reduce_op="mean")

    # sim = Simulation(
    #     config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    # )
    # sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    # ###### 3 - Run the simulation with the GIANT strategy, with attackers ######
    # CONFIG["NUM_ROUNDS"] = NUM_ROUNDS // 2
    # CONFIG["N_BYZANTINE_CLIENTS"] = N_BYZ
    # CONFIG["MODEL_NAME"] = f"GIANT_{CONFIG['N_CLIENTS']}C_{CONFIG['N_BYZANTINE_CLIENTS']}A"

    # strategy = GIANT(CONFIG["MAX_CLIENT_ITERS"])
    # server = Server(CONFIG, reduce_op="mean")

    # sim = Simulation(
    #     config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    # )

    # sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    # ###### 4 - Run the same simulation, but with the FedAvg strategy ######
    # CONFIG["NUM_ROUNDS"] = NUM_ROUNDS
    # CONFIG["MODEL_NAME"] = f"FedAvg_{CONFIG['N_CLIENTS']}C_{CONFIG['N_BYZANTINE_CLIENTS']}A"

    # strategy = FedAvg(CONFIG["MAX_CLIENT_ITERS"])
    # server = Server(CONFIG, reduce_op="mean")

    # sim = Simulation(
    #     config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    # )

    # sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    # ###### 5 - Run the simulation with the GIANT strategy + Median aggregation (MNM) ######
    # CONFIG["NUM_ROUNDS"] = NUM_ROUNDS // 2
    # CONFIG["MODEL_NAME"] = f"GIANT_{CONFIG['N_CLIENTS']}C_{CONFIG['N_BYZANTINE_CLIENTS']}A_MNM"

    # strategy = GIANT(CONFIG["MAX_CLIENT_ITERS"])
    # server = Server(CONFIG, reduce_op="median")

    # sim = Simulation(
    #     config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    # )

    # sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    # ###### 6 - Different simulations for different values of n_byz, to test MNM robustness ######
    n_byzs = [5, 9, 11]
    # n_byzs = [11]
    for n_byz in n_byzs:
        CONFIG["NUM_ROUNDS"] = NUM_ROUNDS // 4
        CONFIG["N_BYZANTINE_CLIENTS"] = n_byz
        CONFIG["MODEL_NAME"] = f"GIANT_{CONFIG['N_CLIENTS']}C_{CONFIG['N_BYZANTINE_CLIENTS']}A_MNM_unif"

        strategy = GIANT(CONFIG["MAX_CLIENT_ITERS"])
        server = Server(CONFIG, reduce_op="median")

        sim = Simulation(
            config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
        )

        sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])
