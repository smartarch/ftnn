package trust40.enforcer.sdq.data;

import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

/**
 * Stores the privacy level for each triple of Subject, Operation, Object
 *
 * It uses internally a 2d array to store the privacy levels
 */
public class PrivacyTable {
    /**
     * Position of SUBJECT
     */
    private final static int SUBJECT = 0;
    /**
     * Position of operation
     */
    private final static int OPERATION = 1;
    /**
     * Position of object
     */
    private final static int OBJECT = 2;
    /**
     * Position
     */
    private final static int PRIVACY_LEVEL = 3;

    private String[][] privacyTable;

    /**
     * Constructor for the Privacy Table
     * @param table 2-dimensional array as a table with 4 columns (subject,operation,object,privacy_level)
     */
    public PrivacyTable (String [][] table){
        privacyTable = Objects.requireNonNull(table);
        if(table.length == 0)
            throw new IllegalArgumentException("Table must be >0");
        if(table[0].length != 4){
            throw  new IllegalArgumentException("Table needs 4 columns");
        }
    }

    /**
     * Returns if exist the privacy level for a given triple
     * @param subject DataObject as the subject
     * @param operation Operation with the operation name, can be allquantified
     * @param object DataObject with the object of the request
     * @return Optional with the privacy level
     */
    public Optional<PrivacyLevel> getPrivacyLevel(DataObject subject, Operation operation, DataObject object){
        return Arrays.stream(privacyTable).filter(e-> {
            Operation checkOperation = Operation.parseOperation(e[OPERATION]);
            boolean operationExpression = checkOperation.equalOperation(operation);
            boolean subjectExpression = e[SUBJECT].equals(subject.getType());
            boolean objectExpression =  e[OBJECT].equals(object.getType());
            return operationExpression && subjectExpression && objectExpression;
        }).map(e-> PrivacyLevel.valueOf(e[PRIVACY_LEVEL].toUpperCase())).findAny();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PrivacyTable that = (PrivacyTable) o;
        return Arrays.deepEquals(privacyTable, that.privacyTable);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(privacyTable);
    }
}
