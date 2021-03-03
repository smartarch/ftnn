package trust40.enforcer.sdq.data;

import java.util.Map;
import java.util.Objects;

/**
 * A dataobject consist of a type and a value. Dataobjects are semantical identical when the type is the same
 */
public class DataObject {
    private String type;
    private String value;

    public DataObject(String type, String value) {
        this.type = Objects.requireNonNull(type);
        this.value = value;
    }
    public DataObject(String type){
        this(type, null);
    }

    /**
     * Creates a DataObject by mapping the string instance (representing a subject/object) to a type
     * @param value String representation of an instance (subject, object)
     * @param datatypeMapping Maping between instance and types (key is instance and value is type level)
     * @return
     */
    public static DataObject mapObject(String value, Map<String, String> datatypeMapping) {
        return new DataObject(datatypeMapping.get(value), value);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DataObject that = (DataObject) o;
        return type.equals(that.type);
    }

    @Override
    public int hashCode() {
        return Objects.hash(type);
    }

    public String getType() {
        return type;
    }

    public String getValue() {
        return value;
    }

    @Override
    public String toString(){
        return "[" + value + ":" + type + "]";
    }
}
