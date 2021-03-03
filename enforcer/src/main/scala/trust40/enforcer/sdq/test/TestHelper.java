package trust40.enforcer.sdq.test;

import trust40.enforcer.sdq.data.PrivacyTable;

public class TestHelper {
    public static PrivacyTable getPrivacyTable(){
        String[][] table = {{"foreman","read(*)","worker","public"},{"foreman","read(*)","machine","sensitive"},{"worker","read(*)","machine","highly_sensitive"}};
        return new PrivacyTable(table);
    }
}
