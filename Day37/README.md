# Cohen–Sutherland Line Clipping Algorithm

The **Cohen–Sutherland Line Clipping Algorithm** is a computer graphics technique used to determine which portions of a line segment are visible within a rectangular clipping window. It helps in optimizing rendering by discarding portions of lines outside the view area.

---

## 📌 How It Works

1. **Divide the 2D Space**  
   The plane is divided into **9 regions** (inside, left, right, top, bottom, and 4 corner regions) based on the rectangular clipping window.

2. **Assign Region Codes**  
   Each endpoint of the line is assigned a **4-bit code** (called the *Outcode*):
   - **Bit 1 (Top)**: Above the window
   - **Bit 2 (Bottom)**: Below the window
   - **Bit 3 (Right)**: Right of the window
   - **Bit 4 (Left)**: Left of the window

3. **Cases**  
   - **Trivial Accept**: If both endpoints have code `0000` → Entire line is inside.  
   - **Trivial Reject**: If bitwise AND of codes ≠ `0000` → Entire line is outside.  
   - **Partial Visibility**: Else, line intersects the boundary — clip step by step.

---

## 🪄 Simple Analogy

Imagine a **photo frame** on your wall:  
- If a string is **completely inside** the frame, you see all of it (accept).  
- If it’s **completely outside**, you ignore it (reject).  
- If part of the string crosses the frame, you trim the extra portions so it fits perfectly inside (clip).

---

## 🧠 Advanced Analogy

Think of a **VIP event** with a rectangular stage area:  
- Each guest (line endpoint) has a **4-bit pass** indicating their position relative to the stage (top, bottom, left, right).  
- If both guests are already inside the stage area, no security check is needed.  
- If both are far away in the same wrong direction (e.g., both behind the stage), they’re denied entry.  
- If one is inside and the other outside, security escorts the outside guest to enter exactly at the boundary — no more, no less.

---

## 🔢 Pseudocode

```plaintext
Assign region codes to both endpoints
While True:
    If both codes are 0000 → ACCEPT
    Else if bitwise AND ≠ 0 → REJECT
    Else:
        Select an endpoint outside the window
        Find intersection with the clipping boundary
        Replace that endpoint with the intersection point
        Update its region code
