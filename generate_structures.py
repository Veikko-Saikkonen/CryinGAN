import torch
from models import Generator
import os
from ase.io import read
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_generator', type=str, default='', help='path to generator model to be loaded')
    parser.add_argument('--n_struc', type=int, default=10, help='number of structures to generate')
    parser.add_argument('--ref_struc', type=str, default='', help='path to reference structure (.extxyz file)')
    parser.add_argument('--latent_dim', type=int, default=64, help='number of random numbers used for generator input')
    parser.add_argument('--gen_channels_1', type=int, default=128, help='number of channels after the first layer of the generator')
    parser.add_argument('--save_dir', type=str, default='./fake_structures', help='directory to save generated structures')
    
    args = parser.parse_args()
    
    # Load generator
    generator = Generator(args)
    print("Loading generator...")
    generator.load_state_dict(torch.load(args.load_generator))
    print("=> Loaded '{}'.".format(args.load_generator))
    generator.eval()
    
    # Generate fake coordinates
    z = torch.randn(args.n_struc, args.latent_dim)
    fake_coords = generator(z).detach()
    
    # Create save directory
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Save structures by replacing the coordinates of the structure template with the generated coordinates
    struc_template = read(args.ref_struc, index=0, format='extxyz')    
    for i in range(len(fake_coords)):
        coords = fake_coords[i][0]
        struc = struc_template.copy()
        struc.set_scaled_positions(coords)
        fname = args.save_dir + '/POSCAR_fake_' + str(i)
        struc.write(fname, format='vasp', direct=True)


if  __name__ == '__main__':
    main()

