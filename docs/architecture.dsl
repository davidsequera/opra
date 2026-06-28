workspace {
    !identifiers hierarchical

    model {

        process_analyst = person "Process Analyst" "Designs, runs, and evaluates business process simulations."

        opra_system = softwareSystem "Process Simulation & RL Optimization" {
            description "Business process simulation and optimization System. With the use reinforcement learning agents to predict activity and resource pair with Top-K / Top-P constrained action spaces."


            /////////////////////////////////////////////////////////////
            // ENVIRONMENT
            /////////////////////////////////////////////////////////////
            environment = container "Simulation Environment" {
                description "Environment connects the agent with the simulator, manages the decision boundary, and handles reward calculation."
                technology "Gymnasium, Python"
                tags "Environment"


                state_representation = component "State Representation" "Encodes the current state of the simulation for the agent."
                mask = component "Mask Actions" "Applies top-k, top-p, and minimum terminal probability (p_end^min) filters to the activity mask, and skill-based filtering to the resource mask, restricting the action space to a process-consistent and feasible subset."
                action_selector = component "Action Selector" "Selects an action (Activity, Resource) from the masked action space based on agent predictions."
                reward = component "Calculate Reward" "Computes the reward signal based on simulation outcomes and performance metrics."
                update_env = component "Update Environment" "Updates the environment state based on executed actions and simulation feedback."
            }

            simulator = container "Discrete-Event Simulator" {
                description "Core simulation engine handling event scheduling and execution."
                technology "Python, SimPy"
                tags "Simulator"

                // Components
                routing_policy = component "Routing Policy" "Determines case routing based on process model."
                processing_time_policy = component "Processing Time Policy" "Predicts activity durations from time models."
                waiting_time_policy = component "Waiting Time Policy" "Calculates waiting times based on resource availability." 
                arrival_policy = component "Arrival Policy" "Generates case arrivals based on arrival models."
                calendar_policy = component "Calendar Policy" "Applies calendar constraints to scheduling."
                resource_policy = component "Resource Allocation Policy" "Allocates resources to activities based on availability and policies."
            }

            /////////////////////////////////////////////////////////////
            // AGENT
            /////////////////////////////////////////////////////////////
            agent = container "RL Agent" {
                description "Learns activity-resource allocation policies."
                technology "Python, PyTorch"
                tags "RL Agent"

                activities_policy = component "Activity Policy Network" "Predicts activity selection probabilities."
                resource_policy = component "Resource Policy Network" "Predicts resource allocation probabilities."
                learn = component "Learn" "Updates policy/value networks."
                save_loss = component "Save Loss" "Persists training loss."
            }

            /////////////////////////////////////////////////////////////
            // EVALUATOR
            /////////////////////////////////////////////////////////////
            evaluator = container "Evaluator" {
                description "Evaluates simulation quality and learning behavior."
                technology "Python"
                tags "Evaluator"

                calc_perf = component "Calculate Simulation Performance" "Aggregates KPIs."
                cycle_time = component "Cycle Time" "Computes end-to-end process duration."
                cf_similarity = component "Control-Flow Similarity" "Compares simulated vs real traces."
                loss_evolution = component "Agent Loss Evolution" "Tracks learning stability."
            }


            /////////////////////////////////////////////////////////////
            // INITIALIZER
            /////////////////////////////////////////////////////////////
            initializer = container "Initializer" {
                description "Bootstraps the simulation, process model, and predictive models."
                technology "Python"
                tags "Initializer"

                discover_routing_policy = component "Discover Routing Policy" "Discovers routing policy from event log."
                discover_processing_time_policy = component "Discover Processing Time Policy" "Discovers processing time policy from event log."
                discover_waiting_time_policy = component "Discover Waiting Time Policy" "Discovers waiting time policy from event log."
                discover_arrival_policy = component "Discover Arrival Policy" "Discovers arrival policy from event log."
                discover_calendar_policy = component "Discover Calendar Policy" "Discovers calendar policy from event log"
                discover_resource_policy = component "Discover Resource Allocation Policy" "Discovers resource allocation policy from event log."
                simulation_setup = component "Provides Simulation the Simulation Model" "Loads activities and resources from historical logs."
            }


            /////////////////////////////////////////////////////////////
            // RELATIONSHIPS – INITIALIZER FLOW
            /////////////////////////////////////////////////////////////
            opra_system.initializer.discover_routing_policy -> opra_system.initializer.simulation_setup
            opra_system.initializer.discover_processing_time_policy -> opra_system.initializer.simulation_setup
            opra_system.initializer.discover_waiting_time_policy -> opra_system.initializer.simulation_setup
            opra_system.initializer.discover_arrival_policy -> opra_system.initializer.simulation_setup
            opra_system.initializer.discover_calendar_policy -> opra_system.initializer.simulation_setup
            opra_system.initializer.discover_resource_policy -> opra_system.initializer.simulation_setup

            opra_system.initializer.simulation_setup -> opra_system.simulator.routing_policy
            opra_system.initializer.simulation_setup -> opra_system.simulator.processing_time_policy
            opra_system.initializer.simulation_setup -> opra_system.simulator.waiting_time_policy
            opra_system.initializer.simulation_setup -> opra_system.simulator.arrival_policy
            opra_system.initializer.simulation_setup -> opra_system.simulator.calendar_policy
            opra_system.initializer.simulation_setup -> opra_system.simulator.resource_policy


            /////////////////////////////////////////////////////////////
            // RELATIONSHIPS – AGENT INTERACTION
            /////////////////////////////////////////////////////////////
            opra_system.environment.state_representation -> opra_system.agent.activities_policy
            opra_system.environment.state_representation -> opra_system.agent.resource_policy
            opra_system.agent.activities_policy -> opra_system.environment.mask "Environment masks acitivity from the agent"
            opra_system.agent.resource_policy -> opra_system.environment.mask "Environment masks resource from the agent"
            
            opra_system.environment.mask -> opra_system.environment.action_selector "Environment applies top-k, top-p, and minimum terminal probability filters to the agent's action space"

            opra_system.environment.action_selector -> opra_system.simulator.processing_time_policy "Selected action (Activity, Resource) is executed in the simulator"
            opra_system.environment.action_selector -> opra_system.simulator.waiting_time_policy "Selected action (Activity, Resource) is executed in the simulator"
            opra_system.environment.action_selector -> opra_system.simulator.calendar_policy "Selected action (Activity, Resource) is executed in the simulator"

            opra_system.environment.action_selector -> opra_system.environment.reward "Agent receives reward signal based on action execution and simulation feedback"
            opra_system.environment.action_selector -> opra_system.environment.update_env "Environment updates state based on action execution and simulation feedback"

            opra_system.environment.reward -> opra_system.agent.learn
            opra_system.agent.learn -> opra_system.agent.save_loss


            /////////////////////////////////////////////////////////////
            // RELATIONSHIPS – EVALUATION
            /////////////////////////////////////////////////////////////
            opra_system.evaluator.calc_perf -> opra_system.evaluator.cycle_time
            opra_system.evaluator.cycle_time -> opra_system.evaluator.cf_similarity
            opra_system.evaluator.cf_similarity -> opra_system.evaluator.loss_evolution

            /////////////////////////////////////////////////////////////
            // USER
            /////////////////////////////////////////////////////////////
            process_analyst -> opra_system.initializer "Configures & launches"
            process_analyst -> opra_system.evaluator "Analyzes results"
        }
    }

    views {

        container opra_system "OPRA_Containers" {
            include *
            autolayout lr
        }

        component opra_system.initializer "Initializer_Flow" {
            include *
            autolayout lr
        }

        component opra_system.environment "Environment_Flow" {
            include *
            autolayout lr
        }

        component opra_system.agent "Agent_Flow" {
            include *
            autolayout lr
        }

        component opra_system.evaluator "Evaluation_Flow" {
            include *
            autolayout lr
        }

        styles {
            element "Initializer" {
                background #BFE8E3
            }
            element "Environment" {
                background #F28B82
            }
            element "RL Agent" {
                background #8AB4F8
            }
            element "Evaluator" {
                background #E8A8E0
            }
            element "Decision Boundary" {
                background #FDD663
                color #000000
                border dashed
            }
        }
    }
}
