import streamlit as st
import numpy as np
import pandas as pid
from streamlit_option_menu import option_menu
from qml import qml_alog
from canvas import canva
from chatbot import chat
from PIL import Image

def main():
    with st.sidebar:
        option = option_menu('QML', 
                               ['Quantum Computing', 'Quantum Machine Learning', 'Quantum Neural Networks', 'Code Application', 'QI Bot','References','About Me'],
                               icons=['cpu', 'activity', 'bricks', 'code', 'robot','link','person'],
                               menu_icon='cast', default_index=0)
    # option = st.sidebar.selectbox("Menu",selected)

    if option=="Quantum Computing":
        # Title of the application
        st.title('Introduction to Quantum Computing')

        # Introduction section
        st.header('What is Quantum Computing?')
        st.write("""
        Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, 
        to perform operations on data. Quantum computers are different from binary digital electronic computers based on transistors. 
        Whereas common digital computing requires that the data be encoded into binary digits (bits), each of which is always in one 
        of two definite states (0 or 1), quantum computation uses quantum bits or qubits.
        """)
        st.image(Image.open('./images/MyQC.png'), caption="IBM's Quantum Computer")

        # Qubits section
        st.header('Qubits')
        st.write("""
        Unlike classical bits, which are binary and can be either 0 or 1, qubits can exist simultaneously in multiple states 
        (0 and 1) thanks to superposition. This allows quantum computers to process a vast number of possibilities simultaneously.
        """)
        st.image(Image.open('./images/Qubit.png'), caption="Qubit vs Classical bits")
        # Superposition section
        st.header('Superposition')
        st.write("""
        Superposition allows a qubit to be in a combination of both 0 and 1 states at the same time. A qubit can be described 
        as α|0⟩ + β|1⟩, where α and β are probability amplitudes. The squared magnitudes of these amplitudes give the probabilities 
        of the qubit being in one of the states when measured.
        """)

        # Entanglement section
        st.header('Entanglement')
        st.write("""
        Entanglement is a quantum phenomenon where pairs or groups of particles interact in such a way that the quantum state 
        of each particle cannot be described independently of the others, even when the particles are separated by large distances. 
        Entangled qubits provide exponential growth in computational power with each additional qubit.
        """)
        st.image(Image.open('./images/Entanglement.jpg'), caption="Entanglement")

        # Quantum Gates section
        st.header('Quantum Gates')
        st.write("""
            Quantum gates are fundamental building blocks in the world of quantum computing, analogous to classical logic gates in traditional computers. These gates manipulate qubits through unitary transformations, which are reversible operations that preserve the norm of the quantum states they act upon. This reversible nature is crucial, as it allows quantum gates to facilitate complex computations that can be undone or retraced.

            ### Examples of Quantum Gates:

            1. **Pauli Gates (X, Y, Z):**
            - **X Gate** (or NOT Gate): Flips the state of a qubit from |0⟩ to |1⟩ and vice versa, similar to a classical NOT gate.
            - **Y and Z Gates**: These perform rotations around the Y and Z axes of the Bloch sphere, respectively, altering the phase of the qubit.

            2. **Hadamard Gate (H):**
            - Transforms the basis states |0⟩ and |1⟩ to superpositions, creating states that are equal parts |0⟩ and |1⟩. This gate is essential for creating superposition in algorithms like Shor's and Grover's.

            3. **Controlled Gates (CNOT, CZ):**
            - **CNOT Gate (Controlled-NOT)**: Performs an X gate on the second (target) qubit only when the first (control) qubit is in the state |1⟩. This gate is a staple in creating entanglement between qubits.
            - **CZ Gate (Controlled-Z)**: Applies a Z gate to the target qubit only when the control qubit is |1⟩.

            4. **T and S Gates:**
            - These are phase gates that introduce more nuanced shifts in the phase of a qubit. The T gate, for example, applies a π/4 phase, useful in many quantum algorithms for fine-tuning the quantum state.

            These gates can be combined in a sequence to build quantum circuits capable of performing complex operations. For more detailed information and interactive examples, visit [Quantum-Inspire's guide on qubits](https://www.quantum-inspire.com/kbase/qubits/).
            """)
        st.image(Image.open('./images/Gates.webp'), caption="Quantum Gates Demonstrated")
        # Quantum Interference section
        st.header('Quantum Interference')
        st.write("""
        Quantum algorithms rely on the phenomenon of interference, where probability amplitudes of certain outcomes are enhanced 
        and others are diminished, leading to a higher probability of the desired outcome.
        """)

        # Measurement section
        st.header('Measurement')
        st.write("""
        Measuring a qubit causes it to collapse from its superposition of states to one of the basis states. This characteristic 
        is critical as it ensures that quantum algorithms provide the correct answer.
        """)

        # Conclusion
        st.header('Conclusion')
        st.write("""
        Quantum computing promises significant advancements in various fields by solving problems currently beyond the reach 
        of classical computers. However, the technology faces challenges such as error rates and qubit coherence times that 
        need to be overcome for practical applications.
        """)
        st.image(Image.open('./images/MyQC2.webp'), caption="")

    elif option=="Quantum Machine Learning":
        # Title of the application
        st.title('Quantum Computing in Machine Learning')

        # Introduction section
        st.header('Introduction')
        st.write("""
        Quantum computing and machine learning are two frontier technologies that are increasingly intersecting. Quantum machine 
        learning (QML) explores how quantum computing can accelerate machine learning tasks and, conversely, how machine learning 
        can help solve complex quantum computing problems.
        """)
        image = Image.open('./images/Qml.png')
        # Resize the image while preserving aspect ratio
        max_size = (300, 300)
        image.thumbnail(max_size)

        # Display the image using Streamlit
        st.image(image, caption="Data vs Models")

        # Quantum Speedup in Machine Learning section
        st.header('Quantum Speedup in Machine Learning')
        st.write("""
        QML aims to use quantum algorithms to improve the computational speed and efficiency of machine learning tasks. This includes:
        - **Quantum data processing**: Processing large datasets much faster than classical computers.
        - **Feature selection and dimensionality reduction**: Using quantum algorithms to improve the efficiency of feature extraction.
        - **Training models**: Quantum algorithms can potentially offer speedups in training complex models like deep neural networks.
        """)

        # Quantum Algorithms for Machine Learning section
        st.header('Quantum Algorithms for Machine Learning')
        st.write("""
        Several quantum algorithms are being adapted or developed for machine learning tasks, including:
        - **Quantum versions of k-means clustering and support vector machine**: For faster and potentially more accurate data classification.
        - **Quantum annealing for optimization**: Used in training neural networks and other optimization tasks in machine learning.
        - **Quantum Boltzmann machines**: Leveraging quantum superposition and entanglement to explore solutions intractable for classical computers.
        """)

        # Potential Applications section
        st.header('Potential Applications')
        st.write("""
        The intersection of quantum computing and machine learning could revolutionize several fields:
        - **Drug discovery**: Faster and more accurate simulations of molecular interactions.
        - **Financial modeling**: Quantum-enhanced algorithms for predicting stock market changes and optimizing portfolios.
        - **Autonomous vehicles**: Improved algorithms for real-time decision making and data processing.
        - **Artificial intelligence**: Enhanced capabilities in learning, reasoning, and pattern recognition.
        """)
        st.image(Image.open('./images/QMLIsNextBig.png'), caption="QML is Next Big Thing")

        # Challenges and Future Directions section
        st.header('Challenges and Future Directions')
        st.write("""
        Despite the promising synergy between quantum computing and machine learning, several challenges remain:
        - **Hardware limitations**: Current quantum computers need further development to run complex machine learning algorithms efficiently.
        - **Error rates and qubit coherence**: Improvements are necessary to ensure reliable and consistent outputs from quantum computations.
        - **Algorithm development**: Many quantum machine learning algorithms are theoretical and require significant refinement for practical applications.
        """)
        st.image(Image.open('./images/QMLinPrac.png'), caption="QML in Practice")

        # Conclusion
        st.header('Conclusion')
        st.write("""
        The integration of quantum computing and machine learning is still at an early stage, but it holds substantial promise for 
        the future. As both fields evolve, they are expected to provide significant breakthroughs in computational science and technology.
        """)

    elif option=="Quantum Neural Networks":
        qml_alog()
    elif option=="Code Application":
        canva()
    elif option=="References":
        st.title('References and Important Links')
        st.markdown("""
        **Educational Resources and Research on Quantum Computing and Quantum Machine Learning:**

        - [Bloch Sphere Visualization Tool](https://bloch.kherb.io/)
        An interactive tool to visualize the state of qubits on the Bloch sphere, useful for understanding quantum states and transformations.

        - [Top Resources to Learn Quantum Machine Learning](https://analyticsindiamag.com/top-resources-to-learn-quantum-machine-learning/)
        A curated list of learning resources and courses for those interested in delving into Quantum Machine Learning.

        - [Research on Nearest Neighbor and Fault-Tolerant Quantum Circuits](https://www.semanticscholar.org/paper/Nearest-Neighbor-and-Fault-Tolerant-Quantum-Circuit-Biswal-Bandyopadhyay/65747f464b0acca574f94f5830267884f95d0bc1/figure/0)
        A research paper discussing advancements in designing nearest neighbor and fault-tolerant quantum circuits, pivotal for practical quantum computing.

        - [Quantum Computing Shorts on YouTube](https://youtube.com/shorts/BjAvFcUiwR4?si=5Wuu0rvbpundkNdV)
        A quick video illustrating concepts in quantum computing.

        - [Introduction to Quantum Machine Learning on YouTube](https://youtu.be/zhQItO6_WoI?feature=shared)
        A comprehensive lecture on quantum machine learning.

        - [Quantum Machine Learning Research Paper](https://sensip.engineering.asu.edu/wp-content/uploads/2021/09/REU-2021_QML-Paper-M-Dobson-1.pdf)
        Detailed research paper exploring various facets of quantum machine learning.

        - [MNIST Classification with Streamlit on GitHub](https://github.com/bhupendrak9917/My-AI-Projects/tree/main/MNIST_Streamlit)
        A project demonstrating the application of machine learning to classify MNIST digits using Streamlit.

        - [TensorFlow Quantum CNN Tutorial](https://www.tensorflow.org/quantum/tutorials/qcnn)
        Learn how to apply quantum convolutional neural networks using TensorFlow Quantum.

        - [Qiskit Machine Learning with PyTorch Connector](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html)
         A tutorial on integrating Qiskit's quantum machine learning capabilities with PyTorch.

          *Additional Learning Resources:*

        - [Supervised Quantum Learning Tutorial](https://www.nature.com/articles/s41586-019-0980-2)
            Natures's tutorial on supervised learning techniques in quantum computing.
        - [Supervised Learning with Quantum Computers](https://www.amazon.com/Supervised-Learning-Quantum-Computers-Technology/dp/3319964232)
           A book by Maria Schuld and Francesco Petruccione dives into the emerging field of quantum machine learning. It explores how quantum computers can be leveraged for tasks like prediction and decision-making based on data.
        - [Introduction to Streamlit for Data Science](https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace)
        An introductory guide to using Streamlit to build data science and machine learning tools.

        - [Docker for Data Scientists](https://www.datacamp.com/community/tutorials/docker-data-science)
        DataCamp tutorial that explains how to use Docker, beneficial for replicating and deploying machine learning models and environments.

        - [Quantum Machine Learning for Classification](https://www.nature.com/articles/s41534-021-00396-8)
        Nature article on how quantum machine learning can be applied to classification tasks.

        - [Streamlit Official Documentation](https://docs.streamlit.io/)
        Official documentation for Streamlit, offering comprehensive guides, tutorials, and API references.

        - [Docker Documentation](https://docs.docker.com/)
        The official Docker documentation provides tutorials, resources, and guides to understand and utilize Docker effectively.
        
        """)
    elif option=="QI Bot":
        chat()
    else:
        st.title('About Me')
        # Short Bio
        st.header('Rama Chandra Kashyap Mamidipalli')
        st.write("""
        Experienced and motivated Software Engineer with three years of agile development, a solid background in Full Stack 
        development, proficiency in Marklogic, and up-to-date knowledge of current technologies, including relational database.
        Demonstrated expertise in crafting clean and efficient code, employing analytical, communication skills, problem-solving skills to
        design effective algorithms. Conducted rigorous testing and debugging processes while maintaining meticulous documentation.
        """)
        # Contact
        st.header('Contact Me')
        st.write("""Email: mamidipalli.r@northeastern.edu""")
        st.write("""LinkedIn: [kashyapmrc](https://www.linkedin.com/in/kashyapmrc)""")
        st.write("""Github: [kashyap0729](https://github.com/kashyap0729)""")
        st.write("""Portfolio: [Portfolio](https://kashyap0729.github.io/portfolio)
        """)

        # Licenses
        st.header('Licenses')
        st.write("""
       # MIT License
        Copyright (c) 2024 Rama Chandra Kashyap

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """)  
    
if __name__=='__main__':
    main()