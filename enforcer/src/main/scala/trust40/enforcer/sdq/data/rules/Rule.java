package trust40.enforcer.sdq.data.rules;

import trust40.enforcer.sdq.data.DataObject;
import trust40.enforcer.sdq.data.Operation;

import java.util.Objects;

public abstract class Rule {
    private final DataObject subject;
    private final Operation operation;
    private final DataObject object;

    public Rule(DataObject subject, Operation operation, DataObject object) {
    	if(subject == null)
    		throw new IllegalArgumentException("subject is null");
    	if(operation == null)
    		throw new IllegalArgumentException("subject is null");
    	if(object == null)
    		throw new IllegalArgumentException("subject is null");
        this.subject = subject;
        this.operation = operation;
        this.object = object;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        Rule rule = (Rule) o;
        return subject.equals(rule.subject) && operation.equals(rule.operation) && object.equals(rule.object);
    }

    @Override
    public int hashCode() {
        return Objects.hash(subject, operation, object);
    }

    public final DataObject getSubject() {
        return subject;
    }

    public final Operation getOperation() {
        return operation;
    }

    public final DataObject getObject() {
        return object;
    }

    /**
     * Checks equality on the type Level
     * @param f
     * @return
     */
    public final boolean equalRule(Rule f) {
        if (f == null)
            return false;
        if (this.equals(f))
            return true;
        return f.operation.equalOperation(getOperation()) && subject.getType().equals(f.subject.getType()) && object.getType().equals(f.object.getType());
    }

    @Override
    public String toString() {
        return "[" + subject + " " + operation + " " + object + "]";
    }

}
