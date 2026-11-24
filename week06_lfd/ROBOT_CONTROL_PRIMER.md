# Level 6 Primer: Controlling a 7-DoF Franka Emika Panda Arm
## Operational Space Control (OSC), Coordinate Frames & Manipulation Mechanics

**Module:** COMP0037 (Robotic Systems / Robot Learning)  
**Target Audience:** Level 6 / MSc students with AI/CS backgrounds transitioning from 2-DoF planar arms to 7-DoF 3D spatial manipulators.  
**Companion Files:** `week06_lfd/collect_demos.py`, `week06_lfd/bc.py`

---

## 1. Why Do We Need This Primer?

In **Weeks 03 and 04**, you controlled a 2-DoF planar robot arm ($q_1, q_2$). The arm moved in a flat 2D plane ($x, y$), and its kinematics could be written down using basic high-school trigonometry:
$$x = l_1 \cos(q_1) + l_2 \cos(q_1 + q_2), \quad y = l_1 \sin(q_1) + l_2 \sin(q_1 + q_2)$$

In **Week 06**, you are suddenly handed an industrial **Franka Emika Panda** robot with **7 revolute joints**, operating in a full 3D spatial environment, manipulating physical objects with a two-finger parallel gripper.

If you had to compute 7-DoF Inverse Kinematics (IK), coordinate transformations, and motor torques by hand, it would take half a semester of classical robotics kinematics. 

> [!TIP]
> **The Good News:** You do **not** have to compute 7-DoF kinematics or motor torques.  
> The simulator runs an industrial low-level controller called an **Operational Space Controller (OSC)** that does all the heavy mathematical lifting for you. You only control the hand in **Cartesian 3D space**!

---

## 2. Joint Space vs Operational (Task) Space

Understanding manipulation begins with choosing the right coordinate representation:

```
                              Joint Space (ℝ⁷)
                   [q₁, q₂, q₃, q₄, q₅, q₆, q₇]ᵀ
        (Unintuitive: What does rotating Joint 3 by 0.1 rad do to the hand?)
                                    │
                                    │ Forward Kinematics x = FK(q)
                                    ▼
                           Operational Space (SE(3))
                   Hand Position:    x, y, z ∈ ℝ³
                   Hand Orientation: Quaternions or Roll, Pitch, Yaw
        (Intuitive: "Move the gripper forward 5 cm, descend 10 cm, pinch!")
```

1. **Joint Space ($\mathbf{q} \in \mathbb{R}^7$):**
   - Represents the physical angles of the 7 motor encoders along the robot arm.
   - Controllable directly via motor torques $\tau \in \mathbb{R}^7$.
   - **Problem:** Highly non-linear and non-intuitive. Moving a gripper in a straight vertical line downward requires complex, simultaneous non-linear accelerations across all 7 joints.
2. **Operational / Task Space ($\mathbf{x} \in \mathbb{R}^3, \mathbf{R} \in SO(3)$):**
   - Represents where the robot's "hand" (the **End-Effector**, or EEF) is located in Cartesian coordinates relative to the world frame.
   - For our task, the tabletop coordinate frame is centered near the front of the robot:
     - $+X$: Forward (away from robot base).
     - $+Y$: Left / Right across the table.
     - $+Z$: Upward (height above the table).

---

## 3. The Operational Space Controller (OSC) Explained

How do we tell a 7-joint robot to move its hand straight down?

In Robosuite, the robot is equipped with an **Operational Space Pose Controller (`OSC_POSE`)**:
1. You supply a **desired Cartesian velocity** $\Delta \mathbf{x} \in \mathbb{R}^3$.
2. The controller uses the robot's physical Jacobian matrix $J(\mathbf{q}) = \frac{\partial \mathbf{x}}{\partial \mathbf{q}} \in \mathbb{R}^{6 \times 7}$ and inertia matrix $M(\mathbf{q})$ to compute the exact joint torques required:
   $$\tau = J^T(\mathbf{q}) \Lambda(\mathbf{q}) \left( K_p (\mathbf{x}_{\text{des}} - \mathbf{x}) - K_d \dot{\mathbf{x}} \right) + g(\mathbf{q})$$
   where $\Lambda(\mathbf{q}) = (J M^{-1} J^T)^{-1}$ is the operational space inertia tensor, and $g(\mathbf{q})$ cancels gravity.
3. **The Result:** The robot's hand moves cleanly along the Cartesian vector you specified, while the arm's 7 joints naturally flex and self-coordinate.

---

## 4. The 7-D Action Vector You Control

In `collect_demos.py`, `bc.py`, and `airl.py`, every action $\mathbf{a} \in \mathbb{R}^7$ sent to the robot is bounded in $[-1.0, 1.0]$:

$$\mathbf{a} = \begin{bmatrix} a_0 \\ a_1 \\ a_2 \\ a_3 \\ a_4 \\ a_5 \\ a_6 \end{bmatrix} = \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \\ \Delta \text{roll} \\ \Delta \text{pitch} \\ \Delta \text{yaw} \\ u_{\text{gripper}} \end{bmatrix}$$

| Action Index | Symbol | Allowed Range | Physical Interpretation |
|:---:|:---:|:---:|:---|
| `a[0]` | $\Delta x$ | $[-1.0, 1.0]$ | End-effector translation along the table (Forward / Backward) |
| `a[1]` | $\Delta y$ | $[-1.0, 1.0]$ | End-effector translation along the table (Left / Right) |
| `a[2]` | $\Delta z$ | $[-1.0, 1.0]$ | End-effector vertical translation (Up / Down) |
| `a[3:6]` | $\Delta \text{rot}$ | $[-1.0, 1.0]$ | End-effector 3D rotation velocity. **Set to $[0, 0, 0]$ in our lab** (the gripper stays oriented vertically pointing straight down). |
| `a[6]` | $u_{\text{grip}}$ | $\{-1.0, 1.0\}$ | Binary gripper finger command: **$-1.0$ = Open fingers**, **$+1.0$ = Close/pinch fingers**. |

---

## 5. Deconstructing Task T1: The Proportional Action Controller

In `collect_demos.py`, Task **T1** asks you to implement `expert_action(eef_pos, target_pos, grip)`:

```python
def expert_action(eef_pos, target_pos, grip):
    # 1. Compute difference vector from current hand position to target waypoint
    delta = np.asarray(target_pos) - np.asarray(eef_pos)
    
    # 2. Scale by a characteristic distance (0.05 m = 5 cm) and clip to [-1, 1]
    action = np.zeros(7, dtype=np.float64)
    action[:3] = np.clip(delta / 0.05, -1.0, 1.0)
    
    # 3. Assign gripper command
    action[6] = grip
    return action
```

### Why does this work?
- If the hand is $10\text{ cm}$ away from the target, $\frac{0.10}{0.05} = 2.0 \implies \text{clip} \to 1.0$ (moves at maximum speed).
- As the hand approaches within $5\text{ cm}$, the velocity naturally scales down: at $1\text{ cm}$ error, velocity is $0.20$.
- When the target is reached, $(target - eef) \to 0 \implies \text{velocity} \to 0$ (smooth, gentle arrival without overshoot).

---

## 6. Deconstructing Task T2: The 5-Phase Grasping Machine

Why can't an expert just fly in a straight diagonal line directly to the cube?

> [!WARNING]
> **The Collision Hazard:** If the gripper flies diagonally towards the cube with fingers open, the fingertips will strike the cube from the side, knocking it across the table before the fingers can close.

To guarantee $100\%$ grasp success, manipulation practitioners use a **Finite State Machine (FSM)** with vertical approach waypoints:

```
                                The 5 Grasping Phases
                                
      Phase 0: HOVER                Phase 1: DESCEND              Phase 2: GRASP
    (Cube + [0, 0, 0.12m])        (Cube + [0, 0, 0.00m])        (Hold position, close)
          ┌───┐                         ┌───┐                         ┌───┐
          │   │ (Open)                  │   │ (Open)                  │███│ (Closed)
          └───┘                         └───┘                         └───┘
            ↓                             ↓                             ↓
          [Cube]                        [Cube]                        [Cube]
     Approaches directly           Fingertips straddle           Fingers pinch cube
     above cube                    cube at table height          (wait GRIP_STEPS=40)
     
                                        │
                                        ▼
      Phase 3: LIFT                 Phase 4: HOLD
    (Cube + [0, 0, 0.15m])        (Cube + [0, 0, 0.15m])
          ┌───┐ (Closed)                ┌───┐ (Closed)
          │███│                         │███│
          [Cube] (Lifted!)              [Cube] (Stationary hold)
            ↑
     Cube hoisted                  Latches success (≥ 4 cm)
     vertically                    Episode terminates
```

### The "Live Cube Tracking" Requirement
In `next_action(obs)`, you must update `self.cube_pos = obs["object-state"][:3]` **at every single step**.  
Why? When the two parallel gripper fingers pinch the cube in Phase 2, friction and contact normal forces inevitably nudge the cube by a few millimeters. If your controller rigidly aims for the initial $t=0$ position, the robot lifts slightly off-center and drops the object!

---

## 7. The 12-Dimensional Observation Space for Behavioral Cloning

In `bc.py` and `airl.py`, the neural network maps a 12-dimensional vector $\mathbf{x} \in \mathbb{R}^{12}$ to the 7D action $\mathbf{a} \in \mathbb{R}^7$:

$$\mathbf{x} = \begin{bmatrix}
\mathbf{x}_{\text{eef}} & (3\text{ floats: } X, Y, Z\text{ end-effector position}) \\
\mathbf{q}_{\text{eef}} & (4\text{ floats: } W, X, Y, Z\text{ end-effector quaternion orientation}) \\
\mathbf{q}_{\text{grip}} & (2\text{ floats: left and right finger joint displacement}) \\
\mathbf{x}_{\text{cube}} & (3\text{ floats: } X, Y, Z\text{ cube center position})
\end{bmatrix} \in \mathbb{R}^{12}$$

### The Supervised Learning Problem
Your MLP policy $\pi_\theta(\mathbf{x}): \mathbb{R}^{12} \to \mathbb{R}^7$ simply learns a smooth non-linear mapping:
$$\pi_\theta(\text{hand position, gripper opening, cube position}) \approx \text{expert velocity command } \mathbf{a}^*$$

By understanding the physical meaning of these 12 inputs and 7 outputs, you can immediately interpret why the policy succeeds, why it suffers from covariate shift when nudged, and why it behaves as an aggressive, memoryless reactive controller.

