# Deinterlacing

Resonance-scanning microscopes generate pixel-streams that most microscopy software 
organize by resonance-scanning cycles (see explanation). Rather than immediately 
organizing pixels into individual lines, pixels are collected through the complete 
oscillation of the resonant scanner. For a 256 x 256 pixel image, this generates 128
512-pixel lines. To form the final image, each line is split down the center to isolate
the 'forward' and 'backward'-scanned data. The backward scanned data is then inverted
and concatenated to the 'forward' scanned data to form two properly oriented left-right
lines. The exact center of the data is usually defined manually within software or
estimated in real-time using software algorithm. However, variations related to 
temperature, drive, and signal-to-noise often result in images with interlacing 
artifacts. Fortunately, these artifacts can be removed following acquisition using
deinterlacing algorithms.


.. admonition Explanation: Resonance Scanning

Resonance scanning microscopes achieve fast image acquisition by pairing a 
slow-galvometric mirror with a fast-resonance scanning mirror: the fast-resonant 
scanning mirror rapidly scans a single-axis of the field-of-view 
(i.e., a horizontal line), while the slow-galvometric mirror moves the line along a 
second-axis. These resonant scanners scanning mirrors achieve 
their rapid motion by vibrating at a fixed frequency in response to applied voltage.
While galvometric scanning mirrors can be tightly-controlled using servos, resonant 
scanning mirrors lack such granular control.  They are extremely under-dampened 
harmonic oscillators: while their frequency is tightly distributed, they are prone to 
large variations in amplitude given very small deviations in their drive. Sychronizing 
the instantaneous angle of the resonance scanner is extremely difficult. Further,
resonance scanning mirrors also display a smooth cosinusoidal velocity through their  
range-of-motion--the scanner moves slower near the limits of its range--that further 
complicates synchronization to an external frequency or other means of fine control. 
The entire microscope is typically aligned to the motion of the resonance scanner.
The slow galvometer is typically synchronized to the approximate'turn-around' 
point of the resonance scanner, where its velocity is closest to (or ideally at) zero.
This is exact synchronization point is usually determined empirically.


    .. centered:: Cycle of a Resonant Scanning Mirror

    .. image:: scan_sync.png
        :width: 800
