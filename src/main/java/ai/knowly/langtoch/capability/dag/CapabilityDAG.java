package ai.knowly.langtoch.capability.dag;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.reflect.TypeToken;
import java.util.*;

/** Class representing a directed acyclic graph (DAG) of capabilities. */
@AutoValue
public abstract class CapabilityDAG {
  public static CapabilityDAG create() {
    return new AutoValue_CapabilityDAG(
        new HashMap<>(),
        ArrayListMultimap.create(),
        new HashMap<>(),
        ArrayListMultimap.create(),
        new HashMap<>());
  }

  abstract HashMap<String, Node<?, ?>> nodes();

  abstract Multimap<String, Object> inputMap();

  abstract HashMap<String, Object> outputMap();

  abstract Multimap<String, String> inDegreeMap();

  abstract HashMap<String, TypeToken<?>> inputTypes();

  /**
   * Add a node to the CapabilityDAG.
   *
   * @param node Node to be added
   * @param inputType Class object representing the input type of the node
   * @param <I> Input type of the node
   * @param <O> Output type of the node
   */
  public <I, O> void addNode(Node<I, O> node, Class<I> inputType) {
    nodes().put(node.getId(), node);
    inputTypes().put(node.getId(), TypeToken.of(inputType));
    for (String outDegree : node.getOutDegree()) {
      inDegreeMap().put(outDegree, node.getId());
    }
  }

  /**
   * Process the CapabilityDAG with the given initial inputs.
   *
   * @param initialInputMap Map of node IDs to their initial input values
   * @return Map of end node IDs to their final output values
   */
  public Map<String, Object> process(Map<String, Object> initialInputMap) {
    for (Map.Entry<String, Object> entry : initialInputMap.entrySet()) {
      setInitialInput(entry.getKey(), entry.getValue());
    }
    List<String> sortedList = topologicalSort();

    for (String id : sortedList) {
      Node<?, ?> node = nodes().get(id);
      Collection<Object> input = inputMap().get(id);
      Object output = processNode(node, input);
      addOutput(id, output);
      for (String outDegree : node.getOutDegree()) {
        addInput(outDegree, output);
      }
    }

    Map<String, Object> result = new HashMap<>();
    for (String id : getEndNodeIds()) {
      result.put(id, outputMap().get(id));
    }
    return result;
  }

  @SuppressWarnings("unchecked")
  private <I, O> O processNode(Node<I, O> node, Collection<Object> input) {
    Iterable<I> typedInput = (Iterable<I>) input;
    return node.process(typedInput);
  }

  public Object getOutput(String id) {
    return outputMap().get(id);
  }

  private List<String> getEndNodeIds() {
    List<String> endNodeIds = new ArrayList<>();
    for (Node<?, ?> node : nodes().values()) {
      if (node.getOutDegree().isEmpty()) {
        endNodeIds.add(node.getId());
      }
    }
    return endNodeIds;
  }

  private void setInitialInput(String id, Object input) {
    TypeToken<?> expectedType = inputTypes().get(id);
    if (!expectedType.isSupertypeOf(input.getClass())) {
      throw new IllegalArgumentException(
          "Input type for node " + id + " does not match the expected type");
    }
    inputMap().put(id, input);
  }

  private void addInput(String id, Object input) {
    inputMap().put(id, input);
  }

  private void addOutput(String id, Object output) {
    outputMap().put(id, output);
  }

  private List<String> topologicalSort() {
    List<String> sorted = new ArrayList<>();
    Queue<String> queue = new LinkedList<>();
    HashMap<String, Integer> inDegrees = new HashMap<>();
    for (Map.Entry<String, Node<?, ?>> entry : nodes().entrySet()) {
      int degree = inDegreeMap().get(entry.getKey()).size();
      inDegrees.put(entry.getKey(), degree);
      if (degree == 0) {
        queue.offer(entry.getKey());
      }
    }

    while (!queue.isEmpty()) {
      String current = queue.poll();
      sorted.add(current);

      for (String outDegree : nodes().get(current).getOutDegree()) {
        int degree = inDegrees.get(outDegree) - 1;
        inDegrees.put(outDegree, degree);
        if (degree == 0) {
          queue.offer(outDegree);
        }
      }
    }

    if (sorted.size() != nodes().size()) {
      throw new IllegalStateException("The graph contains a cycle");
    }
    return sorted;
  }
}
