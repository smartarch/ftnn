package trust40.enforcer.sdq;

import scala.Enumeration;
import trust40.enforcer.sdq.data.DataObject;
import trust40.enforcer.sdq.data.Operation;
import trust40.enforcer.sdq.data.PrivacyLevel;
import trust40.enforcer.sdq.data.PrivacyTable;
import trust40.enforcer.sdq.io.DataTypeMapping;
import trust40.enforcer.sdq.io.PrivacyLoader;
import trust40.enforcer.sdq.data.rules.*;
import trust40.k4case.AllowPermission;
import trust40.k4case.DenyPermission;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

public class DesignTimeDecisionMakerImpl implements DesignTimeDecisionMaker {
    private static final String DEFAULT_PATH_MAPPING = "src/main/scala/trust40/mappingTable.config";
    private static final String DEFAULT_PATH_PRIVACY = "src/main/scala/trust40/privacyLevel.csv";
    private String pathToPrivacyLevels = "";
    private PrivacyTable privacyTable;
    private Map<String, String> datatypeMapping;


    /**
     * @param pathToPrivacyLevels contains the path to the privacy level file
     * @throws IOException
     */
    public DesignTimeDecisionMakerImpl(String pathToPrivacyLevels,String pathToMapping) throws IOException {
        PrivacyLoader privacyLoader = new PrivacyLoader(pathToPrivacyLevels);
        privacyTable = privacyLoader.getPrivacyTable();
        datatypeMapping = new DataTypeMapping(pathToMapping).getTypeMapping();
        this.pathToPrivacyLevels = pathToPrivacyLevels;

    }

    public DesignTimeDecisionMakerImpl() throws IOException {
        this(DEFAULT_PATH_PRIVACY,DEFAULT_PATH_MAPPING);
    }

    @Override
    public List<AllowPermission> validatePolicies(List<AllowPermission> allowPermissions,
                                                  List<DenyPermission> denyPermissions, Map<AllowPermission,List<DenyPermission>> deniedPermissions) {

        // Converts scala permission classes to our rule classes and adds type mapping
        List<AllowRule> allowRules = allowPermissions.stream().map(e -> {
            return new AllowRule(DataObject.mapObject(e.subj(), datatypeMapping),
                    Operation.parseOperation(e.verb()), DataObject.mapObject(e.obj(), datatypeMapping));
        }).collect(Collectors.toList());
        List<DenyRule> denyRules = denyPermissions.stream().map(e -> {
            return new DenyRule(DataObject.mapObject(e.subj(), datatypeMapping),
                    Operation.parseOperation(e.ver()), DataObject.mapObject(e.obj(), datatypeMapping),
                    convertPrivacyLevel(e.lvl()));
        }).collect(Collectors.toList());

        List<AllowRule> deniedAllowRules = new ArrayList<>();
        for(AllowRule rule:allowRules){
            Optional<PrivacyLevel> privacyLevel = privacyTable.getPrivacyLevel(rule.getSubject(),rule.getOperation(),rule.getObject());

            if(!privacyLevel.isPresent())
                continue;

            Collection<DenyRule> deny = getDenyRules(rule,denyRules);
            List<DenyPermission> applicableDenyRules = deny.stream().
                    filter(e -> EnumSet.range(e.getPrivacyLevel(),PrivacyLevel.HIGHLY_SENSITIVE).
                            contains(privacyLevel.get())).
                    map(e->e.getScalaPermission()).collect(Collectors.toList());

            if(!applicableDenyRules.isEmpty()){
                deniedAllowRules.add(rule);
                deniedPermissions.put(rule.getScalaPermission(),applicableDenyRules);
            }
        }
        allowRules.removeAll(deniedAllowRules);

        return allowRules.stream().map(e-> e.getScalaPermission()).collect(Collectors.toList());

    }

    private Collection<DenyRule> getDenyRules(AllowRule allowRule, Collection<DenyRule> denyRules){
        return denyRules.stream().filter(e-> e.equalRule(allowRule)).collect(Collectors.toList());
    }
    private PrivacyLevel convertPrivacyLevel(Enumeration.Value privacyLevel) {
        return PrivacyLevel.valueOf(privacyLevel.toString().toUpperCase());
    }

}
