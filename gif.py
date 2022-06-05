import imageio
path = "data/dynamics_initial_values/plots/"
filenames = []
#for i in range(4, 91,1): 
#    string = path + 'SC_{:0>3}-sc_Phase-fwd.png'.format(i)
#    filenames.append(string)

for i in range(200):
    string = path + f'{i*10+2}.jpg'
    filenames.append(string)

# print(filenames)

with imageio.get_writer(path + 'pi_4pi_4_quench_slow.gif', mode='I', duration = 0.005) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

        