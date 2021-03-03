package trust40.enforcer.sdq.test;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import trust40.enforcer.sdq.data.PrivacyLevel;
import trust40.enforcer.sdq.data.DataObject;
import trust40.enforcer.sdq.data.Operation;
import trust40.enforcer.sdq.data.PrivacyTable;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestPrivacyTable {
    @BeforeAll
    static void init() {
    }
    @Test
    void testExceptions() {
        String[][] tableWrongColumns = new String[1][0];
        String[][] tableWrondRows = new String[0][4];
        assertAll("wrong dimension",
                () -> assertThrows(IllegalArgumentException.class, () -> new PrivacyTable(tableWrongColumns)),
                () -> assertThrows(IllegalArgumentException.class, () -> new PrivacyTable(tableWrondRows))
                );
        assertAll("null",
                () -> assertThrows(NullPointerException.class, () -> new PrivacyTable(null))
        );

    }

    @Test
    void testPrivacyLevelSearch(){
        PrivacyTable privacyTable = TestHelper.getPrivacyTable();
        assertAll("test Privacylevel",
                ()-> assertEquals(PrivacyLevel.PUBLIC, privacyTable.getPrivacyLevel(new DataObject("foreman"),new Operation("read"), new DataObject("worker")).get()),
                ()-> assertEquals(PrivacyLevel.SENSITIVE, privacyTable.getPrivacyLevel(new DataObject("foreman"),new Operation("read"), new DataObject("machine")).get()),
                ()-> assertEquals(PrivacyLevel.HIGHLY_SENSITIVE, privacyTable.getPrivacyLevel(new DataObject("worker"),new Operation("read"), new DataObject("machine")).get())
        );
    }
    @AfterAll
    static void clean() {

    }
}
