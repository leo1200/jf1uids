from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot

input_reader = InputReader("01_case_setup_sod.json", "01_numerical_setup_sod.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocityX", "pressure"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

nrows_ncols = (1,3)
create_lineplot(data_dict, cell_centers, times, nrows_ncols=nrows_ncols, interval=100, static_time=0.2)