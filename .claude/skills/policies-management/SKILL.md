---

name: policies-management

description: Creates or Updates a  Policy, Use when you have to modify any policy of the simulator

---


When creating a policy:

you have to take into account the interfaces that restricts them, remember the idea is to isolate the logic of the policy and provide a common interface for the simulator
in ordet to allow the simualtor to switch between policies without caring about the way they internally work. Remember to follow the software engineering principle of Coupling and Cohesion.