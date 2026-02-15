# Gravity-Aware 3D Load Optimizer 📦⚖️

A Python-based logistics tool designed to solve the **3D Bin Packing Problem** with a focus on real-world physical constraints, weight distribution, and structural stability.

## 🚀 Key Features

* **Physical Stability Validation:** Implements a custom "Support Check" algorithm. An item is only placed if at least **60% of its base** is supported by the items below it, preventing "floating" or unstable configurations.
* **Weight & Payload Management:** Prioritizes container weight limits over pure volume. The optimizer monitors real-time payload and stops packing once the maximum tonnage (e.g., 22 tons for a 20' container) is reached.
* **Dynamic Center of Gravity (CoG):** Automatically calculates the CoG for every container as items are added, providing coordinates (X, Y, Z) to ensure safe handling and transport stability.
* **Multi-Bin Support:** Efficiently handles large inventories by distributing items across multiple containers while maintaining physical constraints for each.
* **Interactive 3D Visualization:** Built with **Streamlit** and **Plotly** to provide an intuitive 3D view of the loaded containers for logistics planners.

## 📐 How It Works



1.  **Preprocessing:** Items are sorted by base area ($Area = L \times W$) to ensure larger, heavier items form a stable base layer.
2.  **Heuristic Packing:** The system uses a Best-Fit approach to find available space.
3.  **Stability Filter:** Every proposed position is passed through an `is_supported()` function to check surface contact.
4.  **Weight Check:** The algorithm reverses placement if the item exceeds the container’s maximum weight capacity.
5.  **CoG Calculation:** The Center of Gravity is updated as a weighted average:  
    $$CoG_{x,y,z} = \frac{\sum (mass_i \times pos_i)}{\sum mass_i}$$

## 🛠️ Technical Stack

* **Language:** Python 3.9+
* **Core Logic:** Modified `py3dbp` heuristic packing algorithm.
* **UI Framework:** Streamlit
* **Visualization:** Plotly (3D Mesh and Scatter charts)
* **Data Handling:** Pandas for Excel/CSV imports

## 🔮 Future Roadmap

* **Strict CoG Constraints:** Automated rejection for loads exceeding specific CoG deviation percentages.
* **TMS/ERP Integration:** API endpoints for seamless data transfer between the optimizer and enterprise systems.
* **Palletization Support:** Optimized packing for standardized pallet dimensions.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
