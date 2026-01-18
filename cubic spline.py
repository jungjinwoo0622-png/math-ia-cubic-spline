import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for file saving
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. Data Input (Final Corrected)
# ==========================================
# Correction: Adjusted Point K from 2.6601 to 2.72 to prevent
# Runge's phenomenon (oscillation) caused by near-vertical slope.
data = np.array([
    [0, 0],       # A
    [0.37, 0.35], # B
    [0.64, 0.27], # C
    [1.06, 0.39], # D
    [1.6, 0.6],   # E
    [2.03, 0.55], # F
    [2.18, 0.58], # G
    [2.34, 0.55], # H
    [2.53, 0.6],  # I
    [2.66, 0.56], # J
    [2.72, 0.37], # K  <-- FIXED (Crucial for stability)
    [3.00, 0.35], # L
    [3.4, 0.45],  # M
    [3.6, 0.65],  # N
    [3.9, 0.7],   # O
    [4.2, 0.88],  # P
    [4.52, 0.83], # Q
    [4.8, 0.89],  # R
    [5, 0.85]     # S
])

x = data[:, 0]
y = data[:, 1]
n = len(x) - 1  # Number of intervals

# ==========================================
# 2. Cubic Spline Matrix Algorithm
# ==========================================
h = np.diff(x) # Step sizes

# Construct Tridiagonal Matrix A
A = np.zeros((n+1, n+1))
A[0, 0] = 1; A[n, n] = 1

for i in range(1, n):
    A[i, i-1] = h[i-1]
    A[i, i] = 2 * (h[i-1] + h[i])
    A[i, i+1] = h[i]

# Construct RHS Vector b
b = np.zeros(n+1)
for i in range(1, n):
    b[i] = 6 * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])

# Solve for Moments (M)
M = np.linalg.solve(A, b)

# Calculate Coefficients a, b, c, d
coeffs = []
for i in range(n):
    a = (M[i+1] - M[i]) / (6 * h[i])
    b = M[i] / 2
    c = (y[i+1] - y[i]) / h[i] - h[i] * (2*M[i] + M[i+1]) / 6
    d = y[i]
    coeffs.append([a, b, c, d])
coeffs = np.array(coeffs)

# ==========================================
# 3. Exact Analytical Integration
# ==========================================
total_volume = 0
total_moment = 0

print("--- Integration Results ---")
for i in range(n):
    P = np.poly1d(coeffs[i]) # P(t) where t = x - xi
    
    # Volume: pi * Integral( P(t)^2 ) dt
    P_sq = P * P
    P_sq_int = np.polyint(P_sq)
    vol = np.pi * (P_sq_int(h[i]) - P_sq_int(0))
    total_volume += vol
    
    # Moment: pi * Integral( (t + xi) * P(t)^2 ) dt
    P_sq_t = np.poly1d([1, 0]) * P_sq # t * P^2
    mom = np.pi * (np.polyint(P_sq_t)(h[i]) - np.polyint(P_sq_t)(0))
    mom += x[i] * vol
    total_moment += mom

# ==========================================
# 4. Final Calculations with Refinement
# ==========================================
v_slit = 0.5 * 1.2 * 1.3 * 0.2
m_slit = 1.8 * v_slit

final_volume = total_volume - v_slit
final_com = (total_moment - m_slit) / final_volume
toppling_angle = np.degrees(np.arctan(0.85 / (5.00 - final_com)))

print(f"Total Raw Volume: {total_volume:.4f} cm^3")
print(f"Final Corrected Volume: {final_volume:.4f} cm^3")
print(f"Final COM (x): {final_com:.4f} cm")
print(f"Height of COM: {5.00 - final_com:.4f} cm")
print(f"Toppling Angle: {toppling_angle:.2f} degrees")

# ==========================================
# 5. Visualization
# ==========================================

# --- Figure 13: Full Profile ---
plt.figure(figsize=(10, 5))
for i in range(n):
    x_range = np.linspace(x[i], x[i+1], 50)
    y_range = coeffs[i][0]*(x_range-x[i])**3 + coeffs[i][1]*(x_range-x[i])**2 + \
              coeffs[i][2]*(x_range-x[i]) + coeffs[i][3]
    plt.plot(x_range, y_range, 'b-', linewidth=2)
plt.scatter(x, y, c='red', zorder=5, label='Data Points (Knots)')
plt.title('Figure 13: Computational Cubic Spline Model of Bishop')
plt.xlabel('x (cm)')
plt.ylabel('Radius y (cm)')
plt.grid(True)
plt.legend()
plt.savefig('figure13_profile.png')
plt.close()

# Figure 14: Comparative Analysis at Neck (Zoomed)
plt.figure(figsize=(8, 6))

idx_L = 11 # Index of point L (3.00)

# [핵심 수정] 범위를 idx_L-4 (Point H)부터 시작하도록 넓혀서 왼쪽 공백 제거
# Indices: 7(H-I), 8(I-J), 9(J-K), 10(K-L), 11(L-M)
plot_range = range(idx_L-4, idx_L+2)

# 1. 빨간색 점선 (Piecewise Linear)
for i in plot_range:
    # 범례 중복 방지
    label = 'Piecewise Linear (Sharp)' if i == idx_L else "_nolegend_"
    # 해당 구간의 시작점(x[i])과 끝점(x[i+1])을 직선으로 연결
    plt.plot([x[i], x[i+1]], [y[i], y[i+1]], 'r--', linewidth=1.5, label=label)

# 2. 파란색 실선 (Cubic Spline)
for i in plot_range: 
    # 해당 구간을 50개 점으로 쪼개서 부드러운 곡선 그리기
    x_range = np.linspace(x[i], x[i+1], 50)
    y_range = coeffs[i][0]*(x_range-x[i])**3 + coeffs[i][1]*(x_range-x[i])**2 + \
              coeffs[i][2]*(x_range-x[i]) + coeffs[i][3]
    
    label = 'Cubic Spline (Smooth)' if i == idx_L else "_nolegend_"
    plt.plot(x_range, y_range, 'b-', linewidth=2.5, label=label)

# Knot Points 표시 (Point K와 L을 강조)
plt.scatter(x[idx_L], y[idx_L], s=100, c='black', zorder=10, label='Knot L (3.00, 0.35)')
plt.scatter(x[idx_L-1], y[idx_L-1], s=50, c='grey', zorder=10) # Point K도 작게 표시

plt.title('Figure 14: Smoothness Verification at Neck (x=3.00)')
plt.xlim(2.6, 3.4) # x축 범위

# y축 범위를 넉넉하게 잡아서 아래로 내려가는 곡선이 잘리지 않게 함
plt.ylim(0.1, 0.6) 

plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.legend(loc='upper left') # 범례 위치 왼쪽 위로 변경 (그래프와 겹치지 않게)
plt.grid(True)
plt.savefig('figure14_comparison.png')
plt.close()

# Figure 15: 3D Solid of Revolution
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
theta = np.linspace(0, 2*np.pi, 30)

for i in range(n):
    x_line = np.linspace(x[i], x[i+1], 20)
    r_line = coeffs[i][0]*(x_line-x[i])**3 + coeffs[i][1]*(x_line-x[i])**2 + \
             coeffs[i][2]*(x_line-x[i]) + coeffs[i][3]
    
    X, THETA = np.meshgrid(x_line, theta)
    R, _ = np.meshgrid(r_line, theta)
    
    Y = R * np.cos(THETA)
    Z = R * np.sin(THETA)
    
    ax.plot_surface(X, Y, Z, color='peru', alpha=0.7)

ax.set_title('Figure 15: 3D Surface Reconstruction')
ax.set_xlabel('Height (x)')
ax.set_ylabel('Width (y)')
ax.set_zlabel('Depth (z)')
# Set axis scaling to be equal for realistic proportion
ax.set_box_aspect([5, 1.7, 1.7]) 
plt.savefig('figure15_3d.png')
plt.close()

