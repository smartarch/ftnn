package trust40.enforcer.sdq.data.rules;

import trust40.enforcer.sdq.data.PrivacyLevel;
import trust40.enforcer.sdq.data.DataObject;
import trust40.enforcer.sdq.data.Operation;
import trust40.k4case.DenyPermission;
import java.util.Objects;

public class DenyRule extends Rule {
    private PrivacyLevel privacyLevel;

    public DenyRule(DataObject subject, Operation action, DataObject object, PrivacyLevel privacyLevel) {
        super(subject, action, object);
        if(privacyLevel == null)
        	throw new IllegalArgumentException("Privacylevel can't be null");
        this.privacyLevel = privacyLevel;
    }

    public PrivacyLevel getPrivacyLevel() {
        return privacyLevel;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        if (!super.equals(o))
            return false;
        DenyRule denyRule = (DenyRule) o;
        return privacyLevel.equals(denyRule.privacyLevel);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), privacyLevel);
    }

    @Override
    public String toString() {
    	return "[" + this.getSubject() + " " + this.getOperation() + " " + this.getObject() + " " + getPrivacyLevel() + "]";
    }

    /**
     * Converts the DenyRule to a {@link DenyPermission} for the scala application
     * @return DenyPermision
     */
    public DenyPermission getScalaPermission(){
        return new DenyPermission(getSubject().getValue(),getOperation().toString(),getObject().getValue(), trust40.enforcer.tcof.PrivacyLevel.withName(getPrivacyLevel().toString()));
    }
}
