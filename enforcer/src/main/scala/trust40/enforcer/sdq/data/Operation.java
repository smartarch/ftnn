package trust40.enforcer.sdq.data;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Describes an operation (verb in scala parts) with parameters
 *
 * An operation can be allQuantified (parameter: *)
 */
public class Operation {
    private final String name;
    private final String[] parameters;
    private boolean allQuantified;

    public Operation(String name, String... parameters) {
        this.name = Objects.requireNonNull(name);
        this.parameters = parameters;
        allQuantified = Arrays.stream(parameters).anyMatch(e -> e.equals("*"));
    }

    /**
     * Creates an operation based on the verb in scala part.
     * Example: "read(*)" will be parsed to a operation with name read and allquantification for parameters
     * @param verb String from Ensembles with the action/verb
     * @return Operation for verb
     */
    public static Operation parseOperation(String verb) {
        if(verb.contains("(")) {
            String operation = verb.substring(0, verb.indexOf('('));
            String[] parameters = verb.substring(verb.indexOf('(') + 1,verb.indexOf(')')).split(",");
            return new Operation(operation, parameters);
        }
        return new Operation(verb);
    }

    public String getName() {
        return name;
    }

    public String[] getParameters() {
        return parameters;
    }

    public boolean isAllQuantified(){
        return allQuantified;
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Operation operation = (Operation) o;
        return name.equals(operation.name) &&
                Arrays.equals(parameters, operation.parameters);
    }

    /**
     * Checks whether the operation is the same regarding the allquantification
     * @param o Operation to check
     * @return true if the operations are identical (including parameters) or the same operation with allquantified parameters
     */
    public boolean equalOperation(Operation o){
        if (o.equals(this))
            return  true;
        return o.name.equals(this.name) && (this.allQuantified || o.allQuantified);
    }
    @Override
    public int hashCode() {
        int result = Objects.hash(name);
        result = 31 * result + Arrays.hashCode(parameters);
        return result;
    }

    @Override
    public String toString() {
        return name  + (parameters.length > 0?  Arrays.stream(parameters).collect(Collectors.joining(",","(",")")): "");
    }
}
