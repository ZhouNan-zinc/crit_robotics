import math
import numpy as np
import csv
import logging
from tqdm import tqdm

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=["small","big"])
    parser.add_argument("velocity", type=float, nargs='?', default=None)
    parser.add_argument("--gravity", type=float, default=1.01325e5)
    parser.add_argument("--temperature", type=float, default=300.)
    parser.add_argument("--pressure", type=float, default=1.01325e5)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.velocity is None:
        logging.warning("Ballestic velocity not set, assuming from ballestic type. This could cause potential danger.")
        args.velocity = 23.3 if args.type == "small" else 15.6

    if args.type == "small":
        args.d = 16.8 * 1e-3
        args.m = 3.2 * 1e-3
    else:
        args.d = 42.5 * 1e-3
        args.m = 44.5 * 1e-3
    
    if args.output is None:
        args.output = f"ballestic_table_{args.type}_at_{args.velocity}m/s"
    return args

UNIFORM_PRESSURE = 1.01325e5
UNIFORM_TEMPERTURE = 273.15
UNIFORM_DYNAMIC_VISCOSITY = 1.7894e-5

def main(args) -> None:
    print("--------- Parameters ---------")
    dynamic_viscosity = UNIFORM_DYNAMIC_VISCOSITY * (args.temperature / 288.15)**1.5 * (288.15+110.4)/(args.temperature + 110.4)
    logging.info(f"dynamic viscosity: {dynamic_viscosity}")

    air_density = 1.293 * (args.pressure / UNIFORM_PRESSURE) * (UNIFORM_TEMPERTURE / args.temperature)
    logging.info(f"air density: {air_density}")

    re = air_density * args.v * args.d / dynamic_viscosity
    logging.info(f"Reynolds number: {re}")

    cd = 24/re + (2.6*(re/5.0))/(1+(re/5.0)**1.52) + (0.411*(re/2.63e5)**(-7.94))/(1+(re/2.63e5)**(-8.00)) + (0.25*(re/1e6))/(1+(re/1e6))
    logging.info(f"air drag coefficient: {re}")
    
    A = np.pi * (args.d/2)**2
    logging.info(f"wind area: {A}")

    f = cd * (math.pi * (args.d**2) / 4) * air_density * (args.v**2) / 2
    logging.info(f"resistence: {A}")

    a = f / args.m
    logging.info(f"accel: {a}")

    print("")
    
    for angle in np.arange(1.25, -1.25, 0.01):
        # 打表，获得从目标位置到角度+时间的映射
        pass

    # 写 CSV
    with open(args.output, 'w', newline='') as f:
        # DO something
        pass

    logging.info(f"All results can be found in {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)