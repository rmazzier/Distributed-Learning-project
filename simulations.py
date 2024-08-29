"""All simulations run for this project are defined here"""

if __name__ == "__main__":
    from fl_simulator import Server, Simulation, FedAvg, GIANT
    from constants import CONFIG

    # Global parameters, constant through different simulations

    CONFIG["N_CLIENTS"] = 10
    CONFIG["NUM_ROUNDS"] = 50
    CONFIG["MAX_CLIENT_ITERS"] = None

    ###### 1 - Run the simulation with the GIANT strategy, NO ATTACKERS ######
    CONFIG["N_BYZANTINE_CLIENTS"] = 0
    CONFIG["MODEL_NAME"] = "GIANT_10C"

    strategy = GIANT(CONFIG["MAX_CLIENT_ITERS"])
    server = Server(CONFIG, reduce_op="mean")

    sim = Simulation(
        config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    )
    sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    ###### 2 - Run the same simulation, but with the FedAvg strategy ######
    CONFIG["MODEL_NAME"] = "FedAvg_10C"

    strategy = FedAvg(CONFIG["MAX_CLIENT_ITERS"])
    server = Server(CONFIG, reduce_op="mean")

    sim = Simulation(
        config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    )
    sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    ###### 3 - Run the simulation with the GIANT strategy, 4 attackers ######
    CONFIG["N_BYZANTINE_CLIENTS"] = 4
    CONFIG["MODEL_NAME"] = "GIANT_10C_4A"

    strategy = GIANT(CONFIG["MAX_CLIENT_ITERS"])
    server = Server(CONFIG, reduce_op="mean")

    sim = Simulation(
        config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    )

    sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    ###### 4 - Run the same simulation, but with the FedAvg strategy ######
    CONFIG["MODEL_NAME"] = "FedAvg_10C_4A"

    strategy = FedAvg(CONFIG["MAX_CLIENT_ITERS"])
    server = Server(CONFIG, reduce_op="mean")

    sim = Simulation(
        config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    )

    sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])

    ###### 5 - Run the simulation with the GIANT strategy + Median aggregation (MNM) ######
    CONFIG["MODEL_NAME"] = "GIANT_10C_4A_MNM"

    strategy = GIANT(CONFIG["MAX_CLIENT_ITERS"])
    server = Server(CONFIG, reduce_op="median")

    sim = Simulation(
        config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    )

    sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])
