package trust40.enforcer.sdq.data.rules;

import trust40.enforcer.sdq.data.DataObject;
import trust40.enforcer.sdq.data.Operation;
import trust40.k4case.AllowPermission;

/**
 * DTO for allow rules
 */
public class AllowRule extends Rule {


    public AllowRule(DataObject subject, Operation operation, DataObject object) {
        super(subject, operation, object);
    }
    /**
     * Converts the AllowRule to a {@link AllowPermission} for the scala application
     * @return AllowPermission
     */
    public AllowPermission getScalaPermission(){
        return new AllowPermission(getSubject().getValue(),getOperation().toString(),getObject().getValue());
    }
}
