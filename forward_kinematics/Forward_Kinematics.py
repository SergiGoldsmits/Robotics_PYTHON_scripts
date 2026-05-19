import numpy as np

def get_modified_dh_matrix(a_prev, alpha_prev, d, theta):
    """
    Computes the Modified DH Transformation Matrix (Craig / Proximal).
    Sequence: Rx(alpha_prev) -> Tx(a_prev) -> Rz(theta) -> Tz(d)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha_prev)
    sa = np.sin(alpha_prev)

    return np.array([
        [ct,         -st,          0,        a_prev],
        [st * ca,     ct * ca,    -sa,      -d * sa],
        [st * sa,     ct * sa,     ca,       d * ca],
        [0,           0,           0,        1     ]
    ])

def get_standard_dh_matrix(a, alpha, d, theta):
    """
    Computes the Standard DH Transformation Matrix (Classic / Distal).
    Sequence: Rz(theta) -> Tz(d) -> Tx(a) -> Rx(alpha)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct,    -st * ca,     st * sa,    a * ct],
        [st,     ct * ca,    -ct * sa,    a * st],
        [0,      sa,          ca,         d     ],
        [0,      0,           0,          1     ]
    ])

def forward_kinematics(q, dh_params, DH_type, world_frame=np.eye(4)):
    """
    Computes the end-effector pose by chaining DH transformation matrices.
    
    Parameters:
    ------------
    q         : list or np.array
                The current joint positions (variables read from encoders).
    dh_params : list of dicts
                Each dict must contain keys: "a", "alpha", "d".
                Optional key: "offset" (defaults to 0.0 if not provided).
    DH_type   : str
                "MOD_DH" for Craig/Modified, "STD_DH" for Classic/Standard.
    world_frame: np.ndarray (4x4)
                The initial transformation matrix representing the robot base setup.
                
    Returns:
    --------
    T_cumulative : np.ndarray (4x4)
                The final homogeneous transformation matrix of the end-effector.
    """
    if len(q) != len(dh_params):
        raise ValueError(f"Dimension mismatch: Received {len(q)} joint angles for {len(dh_params)} DH rows.")

    T_cumulative = world_frame
    
    for i in range(len(dh_params)):
        # Extract physical dimensions from the parameter dict
        a = dh_params[i]["a"]
        alpha = dh_params[i]["alpha"]
        d = dh_params[i]["d"]
        
        # Calculate full theta: joint_variable (q) + constant_offset
        # .get("offset", 0.0) safely defaults to 0 if no offset is explicitly in the table
        theta_offset = dh_params[i].get("offset", 0.0)
        theta = q[i] + theta_offset
        
        # Branch structurally based on the convention requested
        if DH_type == "MOD_DH":
            A_i = get_modified_dh_matrix(a, alpha, d, theta)
        elif DH_type == "STD_DH":
            A_i = get_standard_dh_matrix(a, alpha, d, theta)
        else:
            raise ValueError("Invalid DH_type specified. Use 'MOD_DH' or 'STD_DH'.")
        
        # Post-multiply to chain matrices downstream
        T_cumulative = T_cumulative @ A_i
        
    return T_cumulative

# =====================================================================
# VERIFICATION EXAMPLE: UR5 Nominal Parameters Under Both Conventions
# =====================================================================
if __name__ == "__main__":
    # Test joint configuration: Robot arm bent at shoulder and elbow
    q_test = [0, -np.pi/2, -np.pi/2, 0, 0, 0] 

    # 1. UR5 Table configured for MODIFIED DH (Craig)
    # Notice that offsets can be omitted if they are completely zero.
    ur5_mod_params = [
        {"a": 0,        "alpha": 0,          "d": 0.089159}, 
        {"a": 0,        "alpha": np.pi/2,    "d": 0},        
        {"a": -0.425,   "alpha": 0,          "d": 0},        
        {"a": -0.39225, "alpha": 0,          "d": 0.10915},  
        {"a": 0,        "alpha": np.pi/2,    "d": 0.09465},  
        {"a": 0,        "alpha": -np.pi/2,   "d": 0.0823}    
    ]

    # 2. UR5 Table configured for STANDARD DH (Classic)
    # Notice the distinct placement of the structural offsets 'a' and 'd'
    ur5_std_params = [
        {"a": 0,        "alpha": np.pi/2,    "d": 0.089159, "offset": 0.0},
        {"a": -0.425,   "alpha": 0,          "d": 0.0,      "offset": 0.0},
        {"a": -0.39225, "alpha": 0,          "d": 0.0,      "offset": 0.0},
        {"a": 0,        "alpha": np.pi/2,    "d": 0.10915,  "offset": 0.0},
        {"a": 0,        "alpha": -np.pi/2,   "d": 0.09465,  "offset": 0.0},
        {"a": 0,        "alpha": 0,          "d": 0.0823,   "offset": 0.0}
    ]

    # Evaluate poses
    pose_mod = forward_kinematics(q_test, ur5_mod_params, "MOD_DH")
    pose_std = forward_kinematics(q_test, ur5_std_params, "STD_DH")

    # Print results out neatly
    print("==================================================")
    print("MODIFIED DH END-EFFECTOR POSE:")
    print("==================================================")
    print(np.round(pose_mod, 4))
    print(f"Cartesian (XYZ): {np.round(pose_mod[:3, 3], 5)}")

    print("\n==================================================")
    print("STANDARD DH END-EFFECTOR POSE:")
    print("==================================================")
    print(np.round(pose_std, 4))
    print(f"Cartesian (XYZ): {np.round(pose_std[:3, 3], 5)}")