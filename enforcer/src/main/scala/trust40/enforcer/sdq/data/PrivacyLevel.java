package trust40.enforcer.sdq.data;

/**
 * Enum for handling the privacy level
 */
public enum PrivacyLevel {
    /* The order of appearing describes the order of the enumeration */
    PUBLIC("public"), INTERNAL_USE("internal_use"), SENSITIVE("sensitive"), HIGHLY_SENSITIVE("highly_sensitive");
    private final String name;

    PrivacyLevel(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }

}
