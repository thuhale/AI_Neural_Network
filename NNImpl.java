/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public Node outputNode=null;//the output node
	
	public ArrayList<Instance> trainingSet=null;//the training set
	
	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. The other nodes of inputNodes are of type
 	* input. The remaining nodes are of type sigmoid. 
 	*/
	
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}
		
		//bias node from input layer to output
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);
		
		

		
		//Output node
		outputNode=new Node(4);
		
		//Connecting hidden layer nodes with output node
		for(int i=0;i<hiddenNodes.size();i++)
		{
			NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(i),outputWeights[i]);
			outputNode.parents.add(nwp);
		}
	}
	
	/**
	 * Get the output from the neural network for a single instance
	 * 
	 * The parameter is a single instance
	 */
	
	public Double calculateOutputForInstance(Instance inst)
	{
		// TODO: add code here
		//Set input for the inputNode
		for (int i = 0; i < inst.attributes.size(); i++){
			Node node = inputNodes.get(i);
			node.setInput(inst.attributes.get(i));
		}
		
		for (int i = 0; i < hiddenNodes.size(); i ++){
			Node node = hiddenNodes.get(i);
			node.calculateOutput();
		}
		outputNode.calculateOutput();
		return outputNode.getOutput();
	}
	

	
	
	
	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */
	
	public void train()
	{
		for (int i = 0; i < this.maxEpoch; i ++){
			for (int j = 0; j < this.trainingSet.size(); j++){
				Instance inst = this.trainingSet.get(j);
				this.train(inst);
			}
		}
		//this.train(this.trainingSet.get(0));
		
	}
	
	public void train(Instance inst){
		double out = this.calculateOutputForInstance(inst);
		int trainLabel = inst.classValue;
		double delta = (trainLabel*1.0-out) * out * (1-out);
		Double[] outputWeightChanges = new Double[hiddenNodes.size()];
		
		//calculate weight change for each hidden node
		for(int i = 0; i < hiddenNodes.size(); i++){
			Node node = hiddenNodes.get(i);
			double hiddenActivation = node.getOutput();
			double deltaForHidden = learningRate * hiddenActivation * delta;
			outputWeightChanges[i] = deltaForHidden;
		}
		
		//For each hidden node, calculate weight change from input nodes to it
		double[][] hiddenWeightChanges = new double[hiddenNodes.size()][inputNodes.size()];
		
		for(int i=0; i<hiddenNodes.size()-1; i++)
		{	
			double weightFromHiddenToOutput = outputNode.parents.get(i).weight;
			double hiddenActivation = hiddenNodes.get(i).getOutput();
			
			for(int j=0;j<inputNodes.size();j++)
			{	
				Node inputNode = inputNodes.get(j);
				double deltaForInput = learningRate * inputNode.getOutput() * hiddenActivation * (1-hiddenActivation) * weightFromHiddenToOutput * delta;
				hiddenWeightChanges[i][j] = deltaForInput;
			}
			
		}
		
		//Update weight from hidden to output
		for(int i = 0; i < hiddenNodes.size(); i++){
			outputNode.parents.get(i).weight = outputNode.parents.get(i).weight + outputWeightChanges[i];
		}
		
		//Update weight from input to hidden
		for(int i=0; i<hiddenNodes.size()-1; i++)
		{
			Node node = hiddenNodes.get(i);
			for (int j = 0; j < node.parents.size(); j ++){
				node.parents.get(j).weight = node.parents.get(j).weight + hiddenWeightChanges[i][j];
			}
		}
		
	}
	

}