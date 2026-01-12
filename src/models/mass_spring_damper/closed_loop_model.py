
def create_simple_closed_loop_system(plant_params: ModelParametization):
    plant: control.StateSpace = create_mass_spring_damper(plant_params)
    print(plant.poles())
    controller = Controller(
        controller=FullStateLQRController(
            plant=plant,
            Q=10 * np.eye(4),
            R=10 * np.eye(1),
        ),
        inputs=("", ),
        outputs=("", ),
    ).create_controller()
    print(controller)


if __name__ == "__main__":
    create_simple_closed_loop_system(plant_params=ModelParametization(
        m_1=10,
        m_2=5,
        d_1=0.9,
        d_2=0.9,
        k_1=0.4,
        k_2=0.4,
    ))